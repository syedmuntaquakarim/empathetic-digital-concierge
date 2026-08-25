"""
process_videos.py — Video → Tensor Processing Pipeline
=========================================================
Converts your recorded MP4 files + ARKit CSV exports into
windowed .pt tensor files ready for dataset_v3.py.

Full pipeline per clip:
  1. Extract mono 16kHz WAV from MP4 (ffmpeg)
  2. Detect clap sync marker in WAV (energy threshold)
  3. Load ARKit CSV (blendshape values at 60fps)
  4. Align audio and CSV at the clap frame
  5. Slice into 200ms audio windows + corresponding blendshape frames
  6. Save as .pt tensor file with audio_window + curves

Usage:
    # Process all clips in The Dataset folder:
    python process_videos.py --dataset_dir "D:\\The Dataset" --output_dir "D:\\Tensors_v3"

    # Dry run — shows what will be processed, touches nothing:
    python process_videos.py --dataset_dir "D:\\The Dataset" --output_dir "D:\\Tensors_v3" --dry_run

    # Process single folder only:
    python process_videos.py --dataset_dir "D:\\The Dataset\\Joy_Train" --output_dir "D:\\Tensors_v3\\Joy_Train" --single_folder

Requirements:
    pip install numpy scipy tqdm
    ffmpeg must be installed and on PATH (https://ffmpeg.org/download.html)

ARKit CSV format expected:
    - First column: Timecode or Frame (ignored)
    - Columns 1-260: CTRL_expr blendshape values matching your blendshape_map.py
    - OR columns 1-52: standard ARKit (script will pad to 260 with zeros)
    - Header row required (column names)
    - Values in range 0.0-1.0 (raw, unnormalised)

    Export from UE5 Live Link Face:
        Window → Virtual Production → Live Link
        Select your MetaHuman subject → Record → Export CSV
"""

import os
import sys
import re
import csv
import subprocess
import argparse
import struct
import wave
from pathlib import Path
from typing import Optional

import torch
import numpy as np

try:
    from scipy.io import wavfile
    from scipy.signal import butter, sosfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING] scipy not found. pip install scipy for better WAV loading.")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE      = 16_000        # Hz — model input rate
WINDOW_SAMPLES   = 3_200         # 200ms audio window
STRIDE_SAMPLES   = 1_600         # 100ms stride (50% overlap)
VIDEO_FPS        = 60            # your recording fps
SAMPLES_PER_FRAME = SAMPLE_RATE / VIDEO_FPS   # ≈ 266.67 samples/frame
CURVES_DIM       = 260           # CTRL_expr dimensions

EMOTION_MAP = {
    'joy'       : 0,
    'anger'     : 1,
    'grief'     : 2,
    'neutral'   : 3,
    'amazement' : 4,
}

INTENSITY_FROM_SUFFIX = {
    '110_slow': 1.00,
    '110'     : 1.00,
    '90'      : 0.90,
    '60'      : 0.60,
    '30'      : 0.30,
}

FOLDER_STRUCTURE = [
    ('Neutral_Train',   'neutral'),
    ('Neutral_Test',    'neutral'),
    ('Joy_Train',       'joy'),
    ('Joy_Test',        'joy'),
    ('Anger_Train',     'anger'),
    ('Anger_Test',      'anger'),
    ('Grief_Train',     'grief'),
    ('Grief_Test',      'grief'),
    ('Amazement_Train', 'amazement'),
    ('Amazement_Test',  'amazement'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Metadata parsing
# ─────────────────────────────────────────────────────────────────────────────

def emotion_from_path(path):
    name = os.path.basename(str(path)).lower()
    for emotion_name in sorted(EMOTION_MAP.keys(), key=len, reverse=True):
        if emotion_name in name:
            return emotion_name, EMOTION_MAP[emotion_name]
    return None, None


def intensity_from_path(path, emotion_name):
    if emotion_name == 'neutral':
        return 0.5
    name = os.path.splitext(os.path.basename(str(path)))[0].lower()
    for suffix, val in INTENSITY_FROM_SUFFIX.items():
        if name.endswith(f'_{suffix}') or f'_{suffix}' in name:
            return val
    return 1.0   # default: full intensity


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: WAV extraction from MP4
# ─────────────────────────────────────────────────────────────────────────────

def extract_wav(mp4_path, wav_path, sample_rate=SAMPLE_RATE):
    """
    Extract mono WAV at target sample rate using ffmpeg.
    ffmpeg must be on PATH.
    """
    # ffmpeg handles both .mp4 and .mov natively — no change needed here
    cmd = [
        'ffmpeg', '-y',
        '-i', str(mp4_path),
        '-ac', '1',                      # mono
        '-ar', str(sample_rate),         # 16kHz
        '-acodec', 'pcm_s16le',          # 16-bit PCM
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {mp4_path}:\n{result.stderr.decode()}")
    return wav_path


def load_wav(wav_path):
    """Load WAV file, return (audio_array float32 -1..1, sample_rate)."""
    if HAS_SCIPY:
        sr, data = wavfile.read(str(wav_path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32:
            pass
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    else:
        # Fallback: pure Python wave module
        with wave.open(str(wav_path), 'rb') as wf:
            sr = wf.getframerate()
            n  = wf.getnframes()
            raw = wf.readframes(n)
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return data, sr


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Clap sync detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_clap_sample(audio, sr, search_sec=3.0, threshold_factor=6.0, min_gap_ms=100):
    """
    Detect the clap transient in audio.

    Strategy:
      - Compute short-time energy in 5ms windows
      - Find first window where energy > threshold_factor × mean energy
      - That window start = clap sample index

    Args:
        audio          : float32 array, mono
        sr             : sample rate
        search_sec     : only search first N seconds (clap is always at start)
        threshold_factor: multiplier over mean energy to detect clap
        min_gap_ms     : ignore any peak within this many ms of start (avoids click artefacts)

    Returns:
        clap_sample (int) — sample index of clap onset
    """
    window_ms   = 5
    window_size = int(sr * window_ms / 1000)
    search_n    = int(sr * search_sec)
    min_gap     = int(sr * min_gap_ms / 1000)

    segment = audio[:search_n]
    n_windows = len(segment) // window_size
    energies  = np.array([
        np.mean(segment[i*window_size:(i+1)*window_size] ** 2)
        for i in range(n_windows)
    ])

    mean_energy = energies.mean()
    threshold   = threshold_factor * mean_energy

    min_gap_windows = min_gap // window_size

    for i in range(min_gap_windows, len(energies)):
        if energies[i] > threshold:
            clap_sample = i * window_size
            return clap_sample

    # Fallback: no clap detected — assume clip starts at sample 0
    print("    [WARNING] No clap detected — assuming sync at sample 0")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: CSV loading
# ─────────────────────────────────────────────────────────────────────────────

def load_arkit_csv(csv_path):
    """
    Load ARKit/CTRL_expr blendshape CSV.

    Returns:
        curves: np.array shape (n_frames, CURVES_DIM) float32
        fps:    float (inferred from Timecode column if present, else VIDEO_FPS)

    Handles:
        - 260-column CTRL_expr CSV (direct model input)
        - 52-column standard ARKit CSV (padded to 260 with zeros)
        - First column = Timecode or Frame index (skipped)
    """
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    # Skip first TWO columns (Frame index + Time_sec) per blendshape_map.py:
    # "CSV cols 0,1 = Frame, Time_sec (dropped) | CSV cols 2-261 → tensor dims 0-259"
    data_cols = header[2:]
    n_cols    = len(data_cols)

    frames = []
    for row in rows:
        if len(row) < 3:
            continue
        values = [float(v) if v.strip() else 0.0 for v in row[2:2+n_cols]]
        frames.append(values)

    curves_raw = np.array(frames, dtype=np.float32)   # (n_frames, n_cols)

    if curves_raw.shape[1] == CURVES_DIM:
        curves = curves_raw
    elif curves_raw.shape[1] == 52:
        # Standard ARKit 52 — pad remaining dims with zeros
        curves = np.zeros((curves_raw.shape[0], CURVES_DIM), dtype=np.float32)
        curves[:, :52] = curves_raw
        print(f"    [INFO] 52-col ARKit CSV padded to {CURVES_DIM} dims")
    elif curves_raw.shape[1] < CURVES_DIM:
        curves = np.zeros((curves_raw.shape[0], CURVES_DIM), dtype=np.float32)
        curves[:, :curves_raw.shape[1]] = curves_raw
    else:
        curves = curves_raw[:, :CURVES_DIM]

    # Infer FPS from timecode if available
    fps = VIDEO_FPS
    if 'timecode' in header[0].lower() and len(rows) > 1:
        try:
            t0 = _parse_timecode(rows[0][0])
            t1 = _parse_timecode(rows[1][0])
            if t1 > t0:
                fps = 1.0 / (t1 - t0)
        except Exception:
            pass

    return curves, fps


def _parse_timecode(tc_str):
    """Parse HH:MM:SS:FF or SS.mmm timecode to seconds."""
    tc_str = tc_str.strip()
    if ':' in tc_str:
        parts = tc_str.split(':')
        if len(parts) == 4:
            h, m, s, f = parts
            return int(h)*3600 + int(m)*60 + int(s) + int(f)/VIDEO_FPS
    try:
        return float(tc_str)
    except ValueError:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Alignment + windowing
# ─────────────────────────────────────────────────────────────────────────────

def align_and_window(audio, clap_sample, curves, fps,
                     sr=SAMPLE_RATE,
                     window_samples=WINDOW_SAMPLES,
                     stride_samples=STRIDE_SAMPLES):
    """
    Align audio and curves at the clap frame, then slice into windows.

    The clap occurs simultaneously in audio (at clap_sample) and in the
    CSV (at frame 0, since you start the face capture right before/at clap).
    After alignment, audio[clap_sample:] corresponds to curves[0:].

    Returns list of dicts: [{'audio_window': Tensor(3200,), 'curves': Tensor(260,)}, ...]
    """
    samples_per_frame = sr / fps

    # Trim audio to start at clap
    audio_synced = audio[clap_sample:]

    n_audio  = len(audio_synced)
    n_frames = len(curves)

    # Total usable length in samples (whichever runs out first)
    max_samples = min(n_audio, int(n_frames * samples_per_frame))

    samples = []
    pos = 0
    while pos + window_samples <= max_samples:
        audio_win = audio_synced[pos:pos + window_samples]

        # Corresponding frame index = midpoint of window
        mid_sample = pos + window_samples // 2
        frame_idx  = int(mid_sample / samples_per_frame)
        frame_idx  = min(frame_idx, n_frames - 1)

        curve_frame = curves[frame_idx]

        samples.append({
            'audio_window': torch.from_numpy(audio_win).float(),
            'curves'      : torch.from_numpy(curve_frame).float(),
        })
        pos += stride_samples

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Process single clip
# ─────────────────────────────────────────────────────────────────────────────

def process_clip(mp4_path, csv_path, output_pt_path, tmp_dir, dry_run=False):
    """
    Full pipeline for one (MP4, CSV) pair → .pt tensor file.
    Returns number of windows produced, or 0 on failure.
    """
    mp4_path = Path(mp4_path)
    csv_path = Path(csv_path)

    emotion_name, emotion_id = emotion_from_path(mp4_path)
    intensity = intensity_from_path(mp4_path, emotion_name)

    print(f"  {mp4_path.name}")
    print(f"    emotion={emotion_name}({emotion_id})  intensity={intensity:.2f}")

    if dry_run:
        print(f"    [DRY RUN] would → {output_pt_path}")
        return -1

    # 1. Extract WAV
    wav_path = Path(tmp_dir) / (mp4_path.stem + '.wav')
    extract_wav(mp4_path, wav_path)

    # 2. Load WAV
    audio, sr = load_wav(wav_path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz. Check ffmpeg -ar flag.")

    # 3. Detect clap
    clap_sample = detect_clap_sample(audio, sr)
    clap_ms     = clap_sample / sr * 1000
    print(f"    clap detected at {clap_ms:.1f}ms (sample {clap_sample})")

    # 4. Load CSV
    curves, fps = load_arkit_csv(csv_path)
    print(f"    CSV: {len(curves)} frames @ {fps:.1f}fps  ({len(curves)/fps:.1f}s)")

    # 5. Align + window
    windows = align_and_window(audio, clap_sample, curves, fps)
    print(f"    → {len(windows)} windows produced")

    if len(windows) == 0:
        print(f"    [WARNING] No windows produced — skipping {mp4_path.name}")
        return 0

    # 6. Save
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    torch.save(windows, output_pt_path)

    # Cleanup temp WAV
    try:
        wav_path.unlink()
    except Exception:
        pass

    return len(windows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Find CSV companion for MP4
# ─────────────────────────────────────────────────────────────────────────────

def find_csv(mp4_path):
    """
    Looks for a CSV file with the same stem as the MP4.
    Checks: same folder, CSV subfolder, parent/CSVs folder.

    Export your CSVs from UE5 Live Link Face and place them in the same
    folder as the MP4 with the same filename:
        Joy_Train/joy_train_s01_30.mp4
        Joy_Train/joy_train_s01_30.csv   ← same stem
    """
    mp4  = Path(mp4_path)
    stem = mp4.stem

    # Live Link Face names the CSV the same as the folder/take name.
    # After you rename it to match the video stem, it sits alongside the video.
    candidates = [
        mp4.parent / f"{stem}.csv",
        mp4.parent / "CSVs" / f"{stem}.csv",
        mp4.parent.parent / "CSVs" / f"{stem}.csv",
        # Also check if the video is .mov but csv has same stem
        mp4.with_suffix('.csv'),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main: batch processing
# ─────────────────────────────────────────────────────────────────────────────

def process_all(dataset_dir, output_dir, tmp_dir, dry_run=False, single_folder=False):
    """
    Walk the dataset folder structure and process all MP4+CSV pairs.
    """
    dataset_dir = Path(dataset_dir)
    output_dir  = Path(output_dir)
    tmp_dir     = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Collect all MP4 files
    if single_folder:
        # Treat dataset_dir itself as the folder to process
        folders = [('', str(dataset_dir))]
    else:
        folders = [
            (folder_name, str(dataset_dir / folder_name))
            for folder_name, _ in FOLDER_STRUCTURE
            if (dataset_dir / folder_name).is_dir()
        ]

    total_clips    = 0
    total_windows  = 0
    missing_csvs   = []

    for folder_name, folder_path in folders:
        # Live Link Face exports .mov; also support .mp4 if manually converted
        mp4_files = sorted(
            list(Path(folder_path).glob('*.mp4')) +
            list(Path(folder_path).glob('*.mov'))
        )
        if not mp4_files:
            continue

        print(f"\n{'─'*55}")
        print(f"  Folder: {folder_name or folder_path}  ({len(mp4_files)} clips)")
        print(f"{'─'*55}")

        out_folder = output_dir / folder_name if folder_name else output_dir
        out_folder.mkdir(parents=True, exist_ok=True)

        iterator = tqdm(mp4_files) if HAS_TQDM else mp4_files
        for mp4_path in iterator:
            csv_path = find_csv(mp4_path)
            if csv_path is None:
                print(f"  [MISSING CSV] {mp4_path.name} — skipping")
                missing_csvs.append(str(mp4_path))
                continue

            out_pt = out_folder / (mp4_path.stem + '.pt')
            try:
                n = process_clip(mp4_path, csv_path, out_pt, tmp_dir, dry_run)
                if n > 0:
                    total_windows += n
                    total_clips   += 1
            except Exception as e:
                print(f"  [ERROR] {mp4_path.name}: {e}")

    print(f"\n{'='*55}")
    print(f"  Processing complete")
    print(f"  Clips processed : {total_clips}")
    print(f"  Total windows   : {total_windows}")
    print(f"  Missing CSVs    : {len(missing_csvs)}")
    if missing_csvs:
        print(f"\n  Clips without CSVs (export these from UE5 Live Link Face):")
        for p in missing_csvs:
            print(f"    {p}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Convert MP4 + ARKit CSV pairs to windowed .pt tensor files'
    )
    p.add_argument('--dataset_dir',    type=str, required=True,
                   help='Root folder containing Emotion_Train / Emotion_Test subfolders')
    p.add_argument('--output_dir',     type=str, required=True,
                   help='Where to save .pt tensor files (mirrors folder structure)')
    p.add_argument('--tmp_dir',        type=str, default='_tmp_wav',
                   help='Temp folder for intermediate WAV files (auto-cleaned)')
    p.add_argument('--dry_run',        action='store_true',
                   help='List what would be processed without doing anything')
    p.add_argument('--single_folder',  action='store_true',
                   help='Treat --dataset_dir as a single flat folder of MP4s')
    p.add_argument('--window_ms',      type=int, default=200,
                   help='Audio window size in milliseconds (default: 200)')
    p.add_argument('--stride_ms',      type=int, default=100,
                   help='Window stride in milliseconds (default: 100, 50%% overlap)')
    args = p.parse_args()

   # Apply window/stride overrides
    WINDOW_SAMPLES = int(SAMPLE_RATE * args.window_ms / 1000)
    STRIDE_SAMPLES = int(SAMPLE_RATE * args.stride_ms / 1000)
    
    print(f"\nEmpatheConcierge V3 — Video Processing Pipeline")
    print(f"  Dataset : {args.dataset_dir}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Window  : {args.window_ms}ms ({WINDOW_SAMPLES} samples)")
    print(f"  Stride  : {args.stride_ms}ms ({STRIDE_SAMPLES} samples)")
    print(f"  Dry run : {args.dry_run}")

    process_all(
        dataset_dir   = args.dataset_dir,
        output_dir    = args.output_dir,
        tmp_dir       = args.tmp_dir,
        dry_run       = args.dry_run,
        single_folder = args.single_folder,
    )
