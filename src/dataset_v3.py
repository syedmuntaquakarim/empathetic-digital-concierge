"""
dataset_v3.py — EmpatheConcierge V3 Dataset
=============================================
Changes from V2:
  - 5 emotions: Joy / Anger / Grief / Neutral / Amazement
  - Intensity scalar parsed from tensor filename (_30/_60/_90/_110/_110_slow)
  - Returns 4-tuple: (audio, curves, emotion, intensity)
  - Folder-aware loading: reads from {Emotion}_Train / {Emotion}_Test structure
  - Backward compatible: falls back gracefully if intensity not in filename

Tensor files are produced by process_videos.py and named:
    joy_train_s01_30.pt         → emotion=joy, sentence=1, intensity=0.30
    anger_train_s03_110_slow.pt → emotion=anger, sentence=3, intensity=1.00
    neutral_train_01.pt         → emotion=neutral, intensity=None → defaults to 0.5
"""

import os
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from blendshape_map import REGIONS

# ─────────────────────────────────────────────────────────────────────────────
# Emotion map — 5 emotions for V3
# ─────────────────────────────────────────────────────────────────────────────

FOCUSED_EMOTIONS = {
    'joy'       : 0,
    'anger'     : 1,
    'grief'     : 2,
    'neutral'   : 3,
    'amazement' : 4,
}

# Intensity scalar from filename suffix
# T4 and T5 both map to 1.0 — the "slow" aspect is already encoded in audio features
INTENSITY_FROM_SUFFIX = {
    '30'      : 0.30,
    '60'      : 0.60,
    '90'      : 0.90,
    '110'     : 1.00,
    '110_slow': 1.00,
}

# Neutral has no intensity gradient — use 0.5 (mid-scale) as default
NEUTRAL_INTENSITY = 0.5


def emotion_from_path(path):
    """Returns (emotion_id, emotion_name) or (None, None) if not a focused emotion."""
    name = os.path.basename(path).lower()
    for emotion_name, eid in FOCUSED_EMOTIONS.items():
        if emotion_name in name:
            return eid, emotion_name
    return None, None


def intensity_from_path(path, emotion_name):
    """
    Parse intensity scalar from tensor filename.
    Looks for _30 / _60 / _90 / _110 / _110_slow suffix patterns.
    Returns float 0.0-1.0.
    """
    if emotion_name == 'neutral':
        return NEUTRAL_INTENSITY

    name = os.path.splitext(os.path.basename(path))[0].lower()

    # Check longest match first (110_slow before 110)
    for suffix, intensity in sorted(INTENSITY_FROM_SUFFIX.items(),
                                    key=lambda x: len(x[0]), reverse=True):
        if name.endswith(f'_{suffix}') or f'_{suffix}_' in name:
            return intensity

    # No suffix found — default to full intensity (conservative assumption)
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Region-aware normalisation (unchanged from V2)
# ─────────────────────────────────────────────────────────────────────────────

BS_DIMS   = list(range(0, 251))
HEAD_DIMS = list(range(251, 257))
FLAG_DIMS = list(range(257, 260))


def compute_region_stats(samples):
    all_curves = torch.stack([s['curves'].float() for s in samples])
    bs_mean  = all_curves[:, BS_DIMS].mean(0)
    bs_std   = all_curves[:, BS_DIMS].std(0).clamp(min=1e-6)
    hp_mean  = all_curves[:, HEAD_DIMS].mean(0)
    hp_std   = all_curves[:, HEAD_DIMS].std(0).clamp(min=1e-6)
    return {
        'bs_mean'    : bs_mean,
        'bs_std'     : bs_std,
        'hp_mean'    : hp_mean,
        'hp_std'     : hp_std,
        'curves_mean': all_curves.mean(0),
        'curves_std' : all_curves.std(0).clamp(min=1e-6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RegionNormDataset(Dataset):
    """
    Returns 4-tuple: (audio, curves, emotion_id, intensity)
    intensity: scalar float tensor (1,) in range [0.0, 1.0]
    """
    def __init__(self, samples, emotion_id, intensity, stats):
        self.samples    = samples
        self.emotion_id = emotion_id
        self.intensity  = intensity    # float scalar, same for all samples in this file
        self.stats      = stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        audio  = s['audio_window'].float()
        curves = s['curves'].float()

        # Z-score audio per sample
        mu  = audio.mean()
        std = audio.std().clamp(min=1e-6)
        audio = (audio - mu) / std

        # Region-aware curve normalisation
        out     = torch.zeros_like(curves)
        out[BS_DIMS]   = (curves[BS_DIMS]   - self.stats['bs_mean'])  / self.stats['bs_std']
        out[HEAD_DIMS] = (curves[HEAD_DIMS]  - self.stats['hp_mean'])  / self.stats['hp_std']
        out[FLAG_DIMS] = 0.0

        emotion_t   = torch.tensor(self.emotion_id, dtype=torch.long)
        intensity_t = torch.tensor([self.intensity], dtype=torch.float32)   # (1,)

        return audio, out, emotion_t, intensity_t


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader builder — two calling modes
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(tensor_dir, batch_size=32, num_workers=2, seed=42,
                      folder_mode=False):
    """
    Build train / val / test DataLoaders.

    Args:
        tensor_dir  : path to directory containing .pt tensor files
        batch_size  : batch size
        num_workers : DataLoader workers
        seed        : random seed for stratified split
        folder_mode : if True, scans {Emotion}_Train / {Emotion}_Test subfolders
                      and uses the _Test folders as the held-out test set.
                      if False (default), uses flat directory with 80/10/10 split
                      (backward compatible with V2 behaviour).

    Returns: (train_loader, val_loader, test_loader, stats)
    Each batch: (audio, curves, emotion, intensity)
                 (B,3200) (B,260) (B,) long  (B,1) float
    """
    if folder_mode:
        return _build_from_folders(tensor_dir, batch_size, num_workers, seed)
    else:
        return _build_flat(tensor_dir, batch_size, num_workers, seed)


def _collect_pt_files(directory):
    """Return sorted list of .pt files in directory, skipping missing dirs."""
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.pt')
    ])


def _load_file(path, stats):
    """Load a .pt file and wrap in RegionNormDataset."""
    eid, ename = emotion_from_path(path)
    if eid is None:
        return None
    intensity = intensity_from_path(path, ename)
    data  = torch.load(path, map_location='cpu', weights_only=False)
    samps = data if isinstance(data, list) else [data]
    return RegionNormDataset(samps, eid, intensity, stats)


def _build_from_folders(tensor_dir, batch_size, num_workers, seed):
    """
    Folder-mode: reads from {Emotion}_Train and {Emotion}_Test subfolders.
    _Train folders → stratified 90/10 train/val split
    _Test folders  → held-out test set (no leakage)

    Expected layout:
        tensor_dir/
            Joy_Train/    joy_train_s01_30.pt  ...
            Joy_Test/     joy_test_s01_30.pt   ...
            Anger_Train/  ...
            ...
    """
    print(f"\n{'='*50}")
    print(f"  V3 Dataset — Folder Mode (5 Emotions + Intensity)")
    print(f"{'='*50}")

    rng = random.Random(seed)

    # ── Discover train files per emotion ────────────────────────────────
    train_files_by_emotion = {}
    test_files             = []

    for emotion_name, eid in FOCUSED_EMOTIONS.items():
        train_folder = os.path.join(tensor_dir, f"{emotion_name.capitalize()}_Train")
        test_folder  = os.path.join(tensor_dir, f"{emotion_name.capitalize()}_Test")

        tr = _collect_pt_files(train_folder)
        te = _collect_pt_files(test_folder)
        train_files_by_emotion[eid] = (emotion_name, tr)
        test_files.extend(te)

        print(f"  {emotion_name:<12} train={len(tr):3d}  test={len(te):3d}")

    # ── Stratified 90/10 train/val split from Train folders ─────────────
    train_files, val_files = [], []
    for eid, (ename, files) in train_files_by_emotion.items():
        rng.shuffle(files)
        n_val = max(1, round(len(files) * 0.10))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    print(f"\n  Split (train folders): {len(train_files)} train / {len(val_files)} val")
    print(f"  Test set (test folders): {len(test_files)} files")

    # ── Stats from training files only ──────────────────────────────────
    print("  Computing region-aware normalisation stats from train split...")
    all_train_samples = []
    for path in train_files:
        data  = torch.load(path, map_location='cpu', weights_only=False)
        samps = data if isinstance(data, list) else [data]
        all_train_samples.extend(samps)

    stats = compute_region_stats(all_train_samples)
    print(f"  BS std range:  {stats['bs_std'].min():.4f} – {stats['bs_std'].max():.4f}")
    print(f"  Head std range: {stats['hp_std'].min():.4f} – {stats['hp_std'].max():.4f}")

    # ── Build datasets ───────────────────────────────────────────────────
    def make_concat(file_list):
        datasets = [d for d in (_load_file(p, stats) for p in file_list) if d is not None]
        return ConcatDataset(datasets) if datasets else None

    train_ds = make_concat(train_files)
    val_ds   = make_concat(val_files)
    test_ds  = make_concat(test_files)

    print(f"  Samples: {len(train_ds):,} train / {len(val_ds):,} val / {len(test_ds):,} test")
    print(f"{'='*50}\n")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        stats,
    )


def _build_flat(tensor_dir, batch_size, num_workers, seed):
    """
    Flat mode: single directory of .pt files, 80/10/10 stratified split.
    Backward compatible with V2. Now includes Amazement + intensity.
    """
    print(f"\n{'='*50}")
    print(f"  V3 Dataset — Flat Mode (5 Emotions + Intensity)")
    print(f"{'='*50}")

    all_pt = sorted([
        os.path.join(tensor_dir, f)
        for f in os.listdir(tensor_dir)
        if f.endswith('.pt')
    ])

    emotion_files = {eid: [] for eid in FOCUSED_EMOTIONS.values()}
    n_skipped = 0
    for path in all_pt:
        eid, _ = emotion_from_path(path)
        if eid is None:
            n_skipped += 1
        else:
            emotion_files[eid].append(path)

    for name, eid in FOCUSED_EMOTIONS.items():
        print(f"  {name:<12} class={eid}  files={len(emotion_files[eid])}")
    print(f"  Skipped: {n_skipped}")

    rng = random.Random(seed)
    train_files, val_files, test_files = [], [], []
    for eid, files in emotion_files.items():
        rng.shuffle(files)
        n       = len(files)
        n_val   = max(1, round(n * 0.10))
        n_test  = max(1, round(n * 0.10))
        n_train = n - n_val - n_test
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    print(f"  Split: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")

    print("  Computing normalisation stats...")
    all_train_samples = []
    for path in train_files:
        data  = torch.load(path, map_location='cpu', weights_only=False)
        samps = data if isinstance(data, list) else [data]
        all_train_samples.extend(samps)

    stats = compute_region_stats(all_train_samples)

    def make_concat(file_list):
        datasets = [d for d in (_load_file(p, stats) for p in file_list) if d is not None]
        return ConcatDataset(datasets) if datasets else None

    train_ds = make_concat(train_files)
    val_ds   = make_concat(val_files)
    test_ds  = make_concat(test_files)

    print(f"  Samples: {len(train_ds):,} train / {len(val_ds):,} val / {len(test_ds):,} test")
    print(f"{'='*50}\n")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset_v3.py <tensor_dir> [--folder]")
        sys.exit(1)

    folder_mode = '--folder' in sys.argv
    tl, vl, tsl, stats = build_dataloaders(
        sys.argv[1], batch_size=8, num_workers=0, folder_mode=folder_mode
    )

    batch = next(iter(tl))
    audio, curves, emotion, intensity = batch

    print(f"audio shape    : {audio.shape}")
    print(f"curves shape   : {curves.shape}")
    print(f"emotion shape  : {emotion.shape}  values: {emotion.unique().tolist()}")
    print(f"intensity shape: {intensity.shape} range: [{intensity.min():.2f}, {intensity.max():.2f}]")
    print(f"curves brows   : {curves[:,:4].mean():.3f} ± {curves[:,:4].std():.3f}  (expect ~N(0,1))")
    print(f"curves head    : {curves[:,251:257].mean():.3f} ± {curves[:,251:257].std():.3f}  (expect ~N(0,1))")
    print(f"curves flags   : {curves[:,257:260].abs().max():.4f}  (expect 0)")
    print("Dataset V3 smoke test passed ✓")
