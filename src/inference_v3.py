import torch
import torchaudio
import numpy as np
import pandas as pd
import argparse
import math
from model_v2 import EmpatheConciergeV3

# Emotion mapping for V3 (5-class space)
EMOTION_MAP = {
    'joy': 0, 'anger': 1, 'grief': 2, 'neutral': 3, 'amazement': 4
}

def run_inference(audio_path, output_csv, emotion_label, intensity_val, ckpt_path, fps=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading V3 Model on {device}...")

    # 1. Initialize Model & Load Weights
    model = EmpatheConciergeV3(num_emotions=5).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"✓ Loaded best_v3.pt (Trained to Val Loss: {ckpt.get('best_val', 'N/A'):.5f})")

    # 2. Process Audio (Bypass torchaudio's broken loader)
    import soundfile as sf
    audio_data, sr = sf.read(audio_path)
    waveform = torch.tensor(audio_data, dtype=torch.float32)
    
    # Ensure mono (Stereo to Mono)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  
    else:
        waveform = waveform.mean(dim=0, keepdim=True) 
        
    # Resample to 16kHz
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

    # 3. Sliding Window Logic
    window_size = 3200  # 200ms at 16kHz
    stride = int(16000 / fps)  # ~533 samples for 30 FPS
    
    # Pad the audio so the first frame maps to the very start of the audio
    pad_len = window_size // 2
    padded_waveform = torch.nn.functional.pad(waveform, (pad_len, pad_len))
    
    # Calculate how many 30 FPS frames we will generate
    num_frames = math.ceil(waveform.shape[1] / stride)
    
    windows = []
    for i in range(num_frames):
        start = i * stride
        end = start + window_size
        window = padded_waveform[:, start:end]
        # Pad the very last window if it's too short
        if window.shape[1] < window_size:
            window = torch.nn.functional.pad(window, (0, window_size - window.shape[1]))
        windows.append(window)
        
    audio_batch = torch.cat(windows, dim=0) # Shape: (num_frames, 3200)
    
    # Apply standard Z-score normalization per window (matches dataset_v3.py)
    mu = audio_batch.mean(dim=1, keepdim=True)
    std = audio_batch.std(dim=1, keepdim=True).clamp(min=1e-6)
    audio_batch = (audio_batch - mu) / std

    print(f"\nAudio processed: Generating {num_frames} frames of animation at {fps} FPS...")

    # 4. Format Condition Tensors
    emotion_idx = EMOTION_MAP[emotion_label.lower()]
    
    # 5. Generate Blendshapes in Mini-batches
    # We process 64 frames at a time so your CPU doesn't run out of RAM
    batch_size = 64
    all_curves = []
    
    print(f"Running Reverse Diffusion for [{emotion_label.upper()}] at {intensity_val}x intensity...")
    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            b_audio = audio_batch[i:i+batch_size].to(device)
            b_emotion = torch.tensor([emotion_idx] * b_audio.shape[0], dtype=torch.long, device=device)
            b_intensity = torch.tensor([[intensity_val]] * b_audio.shape[0], dtype=torch.float32, device=device)
            
            curves = model.sample(b_audio, b_emotion, intensity=b_intensity)
            all_curves.append(curves.cpu().numpy())
            
    # Combine all batches into the final animation sequence
    final_curves = np.concatenate(all_curves, axis=0) # Shape: (num_frames, 260)

    # 6. Export for Unreal Engine 5
    df = pd.DataFrame(final_curves)
    df.to_csv(output_csv, index=False, header=False)
    print(f"✓ Success! {num_frames} frames exported to: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EmpatheConcierge V3 Inference")
    parser.add_argument('--audio', type=str, required=True, help="Path to input .wav file")
    parser.add_argument('--out', type=str, default='ue5_animation.csv', help="Output CSV filename")
    parser.add_argument('--emotion', type=str, required=True, choices=EMOTION_MAP.keys(), help="Target emotion")
    parser.add_argument('--intensity', type=float, default=1.0, help="Emotion intensity (e.g., 0.5 to 1.5)")
    parser.add_argument('--ckpt', type=str, default='best_v3.pt', help="Path to checkpoint")
    parser.add_argument('--fps', type=int, default=30, help="Target framerate for Unreal Engine")
    
    args = parser.parse_args()
    run_inference(args.audio, args.out, args.emotion, args.intensity, args.ckpt, args.fps)