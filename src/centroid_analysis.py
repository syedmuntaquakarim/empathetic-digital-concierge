"""
centroid_analysis.py — Emotion Centroid Separation Analysis
============================================================
Run this locally to evaluate Run 2 checkpoint emotion separation.
Usage: python centroid_analysis.py
"""

import torch
import sys
import os

# ── Paths — edit these if different ──────────────────────────────────────────
CKPT_PATH    = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER\checkpoints_v3\best_v3.pt"
STATS_PATH   = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER\checkpoints_v3\dataset_stats_v3.pt"
TENSOR_DIR   = r"D:\Tensors_v3"
SCRIPT_DIR   = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER"
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

# Load val data
_, val_loader, _, _ = build_dataloaders(
    TENSOR_DIR, batch_size=64, num_workers=0, folder_mode=True
)

EMOTION_NAMES = {v: k.capitalize() for k, v in FOCUSED_EMOTIONS.items()}

# Collect per-emotion predictions
emotion_preds = {eid: [] for eid in range(5)}

with torch.no_grad():
    for batch in val_loader:
        audio, curves, emotion, intensity = batch
        audio     = audio.to(device)
        emotion   = emotion.to(device)
        intensity = intensity.to(device)
        pred = model.sample(audio, emotion, T_steps=10, intensity=intensity)
        for eid in range(5):
            mask = emotion == eid
            if mask.any():
                emotion_preds[eid].append(pred[mask].cpu())

# Compute centroids
centroids = {}
print("\nSamples per emotion in val set:")
for eid in range(5):
    if emotion_preds[eid]:
        all_preds = torch.cat(emotion_preds[eid], dim=0)
        centroids[eid] = all_preds.mean(0)
        print(f"  {EMOTION_NAMES[eid]:<12}: {len(all_preds)} samples")

# Pairwise distances
print("\n=== Emotion Centroid Separation (Run 2 — margin=0.05) ===")
print(f"{'Pair':<35} {'Distance':>12}")
print("─" * 50)

pairs = []
eids  = sorted(centroids.keys())
for i in range(len(eids)):
    for j in range(i+1, len(eids)):
        ei, ej = eids[i], eids[j]
        dist = (centroids[ei] - centroids[ej]).pow(2).mean().sqrt().item()
        pairs.append((f"{EMOTION_NAMES[ei]} ↔ {EMOTION_NAMES[ej]}", dist))

pairs.sort(key=lambda x: -x[1])
for name, dist in pairs:
    marker = "  ← highest" if pairs[0][0] == name else (
             "  ← lowest"  if pairs[-1][0] == name else "")
    print(f"{name:<35} {dist:>12.4f}{marker}")

mean_sep = sum(d for _, d in pairs) / len(pairs)
print("─" * 50)
print(f"{'Mean separation':<35} {mean_sep:>12.4f}")

print(f"\nRun 1 baseline (margin=0.35): mean=0.0754, Anger↔Amazement=0.0821")
print(f"Run 2 result   (margin=0.05): mean={mean_sep:.4f}, ", end="")
anger_amaze = next((d for n,d in pairs if "Anger" in n and "Amazement" in n), 0)
print(f"Anger↔Amazement={anger_amaze:.4f}")

if mean_sep > 0.0754:
    print("✓ Centroid separation IMPROVED with recalibrated margin")
else:
    print("→ Centroid separation similar — improvement in val loss, not geometry")
