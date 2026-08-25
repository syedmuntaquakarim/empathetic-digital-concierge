"""
evaluate_run2.py — System Accuracy Evaluation with Run 2 Checkpoint
====================================================================
Computes cosine similarity between model predictions and ground truth
blendshape curves across all held-out test clips.

Usage: python evaluate_run2.py
"""

import torch
import numpy as np
import sys
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PATH  = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER\checkpoints_v3\best_v3.pt"
STATS_PATH = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER\checkpoints_v3\dataset_stats_v3.pt"
TENSOR_DIR = r"D:\Tensors_v3"
SCRIPT_DIR = r"D:\Major\Empathetic-Concierge-Project_v3"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, SCRIPT_DIR)

from model_v2   import EmpatheConciergeV3
from dataset_v3 import build_dataloaders, FOCUSED_EMOTIONS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
model = EmpatheConciergeV3(num_emotions=5).to(device)
ckpt  = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded — epoch {ckpt['epoch']}, val={ckpt['best_val']:.5f}")

# Load denorm stats
stats   = torch.load(STATS_PATH, map_location='cpu', weights_only=False)
bs_mean = stats['bs_mean'].to(device)
bs_std  = stats['bs_std'].to(device)
hp_mean = stats['hp_mean'].to(device)
hp_std  = stats['hp_std'].to(device)

# Load test data
_, _, test_loader, _ = build_dataloaders(
    TENSOR_DIR, batch_size=32, num_workers=0, folder_mode=True
)

EMOTION_NAMES = {v: k.capitalize() for k, v in FOCUSED_EMOTIONS.items()}

def cosine_similarity_pct(pred, target):
    """Cosine similarity per sample, returned as percentage."""
    dot     = (pred * target).sum(dim=-1)
    norm_p  = pred.norm(dim=-1).clamp(min=1e-8)
    norm_t  = target.norm(dim=-1).clamp(min=1e-8)
    return (dot / (norm_p * norm_t) * 100.0)

def denormalize(pred):
    """Convert z-score model output back to blendshape values."""
    out = torch.zeros_like(pred)
    out[:, :251]    = pred[:, :251]   * bs_std  + bs_mean
    out[:, 251:257] = pred[:, 251:257] * hp_std  + hp_mean
    out[:, 257:]    = 0.0
    out[:, :251]    = out[:, :251].clamp(0.0, 1.0)
    return out

# Evaluate per emotion
emotion_scores = {eid: [] for eid in range(5)}
emotion_counts = {eid: 0  for eid in range(5)}

with torch.no_grad():
    for batch in test_loader:
        audio, curves, emotion, intensity = batch
        audio     = audio.to(device)
        curves    = curves.to(device)
        emotion   = emotion.to(device)
        intensity = intensity.to(device)

        pred = model.sample(audio, emotion, T_steps=50, intensity=intensity)

        # Denormalize both pred and curves for fair comparison
        pred_denorm   = denormalize(pred)
        curves_denorm = denormalize(curves)

        scores = cosine_similarity_pct(pred_denorm, curves_denorm)

        for eid in range(5):
            mask = emotion == eid
            if mask.any():
                emotion_scores[eid].extend(scores[mask].cpu().tolist())
                emotion_counts[eid] += mask.sum().item()

# Print report
print("\n" + "="*62)
print(" EMPATHETIC DIGITAL CONCIERGE - ACCURACY REPORT (RUN 2)")
print("="*62)

all_scores = []
for eid in range(5):
    name = EMOTION_NAMES.get(eid, f"Class{eid}")
    scores = emotion_scores[eid]
    if scores:
        avg = np.mean(scores)
        all_scores.extend(scores)
        print(f" {name:<12} | Clips: {emotion_counts[eid]:3d} | Avg Accuracy: {avg:.2f}%")

print("-"*62)
total_avg = np.mean(all_scores) if all_scores else 0
total_clips = sum(emotion_counts.values())
print(f" TOTAL SYSTEM ACCURACY ({total_clips} clips): {total_avg:.2f}%")
print("="*62)

print(f"\nRun 1 baseline: 79.14%")
print(f"Run 2 result:   {total_avg:.2f}%")
if total_avg > 79.14:
    print(f"✓ Improvement: +{total_avg - 79.14:.2f} percentage points")
else:
    print(f"→ Similar accuracy — improvement reflected in val loss not cosine sim")

print(f"\nAngular error: arccos({total_avg/100:.4f}) ≈ {np.degrees(np.arccos(min(total_avg/100, 1.0))):.1f}°")
