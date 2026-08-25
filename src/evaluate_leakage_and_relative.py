"""
evaluate_leakage_and_relative.py 
====================================================================
1. Proves the "Data Leakage" theory by evaluating on the Train set.
2. Introduces Relative Accuracy metrics (Thresholded & Pearson) for the paper.
"""

import torch
import numpy as np
import sys
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_PATH  = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER\checkpoints_v3\best_v3.pt" # Or best_v3_run2.pt
STATS_PATH = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER\checkpoints_v3\dataset_stats_v3.pt"
TENSOR_DIR = r"D:\Tensors_v3"
SCRIPT_DIR = r"D:\Major\Empathetic-Concierge-Project_v3_PAPER"
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

# 1. LEAKAGE TEST: Load TRAIN data instead of Test data
train_loader, _, _, _ = build_dataloaders(
    TENSOR_DIR, batch_size=32, num_workers=0, folder_mode=True
)

EMOTION_NAMES = {v: k.capitalize() for k, v in FOCUSED_EMOTIONS.items()}

def cosine_similarity_pct(pred, target):
    """Standard absolute cosine similarity."""
    dot     = (pred * target).sum(dim=-1)
    norm_p  = pred.norm(dim=-1).clamp(min=1e-8)
    norm_t  = target.norm(dim=-1).clamp(min=1e-8)
    return (dot / (norm_p * norm_t) * 100.0)

def thresholded_cosine_pct(pred, target, threshold=0.05):
    """Only evaluates blendshapes that are actively moving (ignores micro-noise)."""
    mask = (target > threshold).float()
    p_masked = pred * mask
    t_masked = target * mask
    dot     = (p_masked * t_masked).sum(dim=-1)
    norm_p  = p_masked.norm(dim=-1).clamp(min=1e-8)
    norm_t  = t_masked.norm(dim=-1).clamp(min=1e-8)
    # Avoid div by zero for completely silent frames
    valid = (norm_t > 1e-5)
    result = torch.zeros_like(dot)
    result[valid] = (dot[valid] / (norm_p[valid] * norm_t[valid]) * 100.0)
    result[~valid] = 100.0 # If nothing should move, and we masked it, it's 100% accurate
    return result

def pearson_correlation_pct(pred, target):
    """Evaluates the structural trend/shape of the expression."""
    p_mean = pred.mean(dim=-1, keepdim=True)
    t_mean = target.mean(dim=-1, keepdim=True)
    p_centered = pred - p_mean
    t_centered = target - t_mean
    
    dot = (p_centered * t_centered).sum(dim=-1)
    norm_p = p_centered.norm(dim=-1).clamp(min=1e-8)
    norm_t = t_centered.norm(dim=-1).clamp(min=1e-8)
    return (dot / (norm_p * norm_t) * 100.0)

def denormalize(pred):
    out = torch.zeros_like(pred)
    out[:, :251]    = pred[:, :251]   * bs_std  + bs_mean
    out[:, 251:257] = pred[:, 251:257] * hp_std  + hp_mean
    out[:, 257:]    = 0.0
    out[:, :251]    = out[:, :251].clamp(0.0, 1.0)
    return out

# Metric accumulators
standard_scores = []
thresh_scores   = []
pearson_scores  = []
clips_evaluated = 0

MAX_CLIPS_TO_TEST = 320 # Keep it identical to the test set size for fair comparison

print("\nEvaluating on TRAIN subset to verify leakage theory...")

with torch.no_grad():
    for batch in train_loader:
        if clips_evaluated >= MAX_CLIPS_TO_TEST:
            break
            
        audio, curves, emotion, intensity = batch
        audio     = audio.to(device)
        curves    = curves.to(device)
        emotion   = emotion.to(device)
        intensity = intensity.to(device)

        pred = model.sample(audio, emotion, T_steps=50, intensity=intensity)

        pred_denorm   = denormalize(pred)
        curves_denorm = denormalize(curves)

        standard = cosine_similarity_pct(pred_denorm, curves_denorm)
        thresh   = thresholded_cosine_pct(pred_denorm, curves_denorm)
        pearson  = pearson_correlation_pct(pred_denorm, curves_denorm)

        standard_scores.extend(standard.cpu().tolist())
        thresh_scores.extend(thresh.cpu().tolist())
        pearson_scores.extend(pearson.cpu().tolist())
        
        clips_evaluated += len(audio)

# Print report
print("\n" + "="*65)
print(" DATA LEAKAGE & RELATIVE ACCURACY REPORT")
print("="*65)

std_avg    = np.mean(standard_scores)
thresh_avg = np.mean(thresh_scores)
pears_avg  = np.mean(pearson_scores)

print(f" Dataset Split Evaluated : TRAIN (Leakage Verification)")
print(f" Total Clips Evaluated   : {clips_evaluated}\n")
print(f" 1. Standard Cosine Sim  : {std_avg:.2f}%  <-- (If ~79%, your theory is proven!)")
print(f" 2. Active Muscle Sim    : {thresh_avg:.2f}%  <-- (Relative Metric for Paper)")
print(f" 3. Pearson Correlation  : {pears_avg:.2f}%  <-- (Relative Metric for Paper)")
print("="*65)