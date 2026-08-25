"""
train_v3.py — EmpatheConcierge V3 Training
============================================
Losses: Weighted MSE (AWMSE, inside model) + Contrastive + Velocity + Vol-Stab
Changes from V2:
  - num_emotions 4 → 5  (added Amazement)
  - contra_margin default 0.30 → 0.35  (wider gap for 5-class space)
  - Added velocity_loss: penalises unnatural frame-to-frame jitter
  - Added vol_stab_loss: suppresses motion during quiet audio segments
  - Checkpoint interval 10 → 25 epochs
  - All filenames renamed v2 → v3
  - History tracks vel + vol_stab loss
  - Two extra plot panels for new losses

Usage:
  python train_v3.py --tensor_dir "D:\\Dataset\\Tensors_v3" --epochs 150
"""

import os, argparse, contextlib
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from model_v2   import EmpatheConciergeV3          # thin subclass added at bottom of model_v2.py
from dataset_v3 import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def contrastive_loss(pred_curves, emotion_ids, margin=0.35):
    """
    Vectorised contrastive loss.
    Pulls same-emotion predictions together, pushes different-emotion apart.
    Margin raised 0.30→0.35 for 5-class latent space (more classes = more
    crowding, wider margin needed to keep emotions geometrically separated).
    """
    bs   = pred_curves[:, :251]                          # blendshapes only, exclude head dims
    B    = bs.shape[0]
    diff = bs.unsqueeze(1) - bs.unsqueeze(0)             # (B, B, 251)
    dist = diff.pow(2).mean(dim=-1)                      # (B, B)
    same = (emotion_ids.unsqueeze(1) == emotion_ids.unsqueeze(0)).float()
    eye  = torch.eye(B, device=bs.device)
    same = same * (1 - eye)
    diff_m = (1 - same) * (1 - eye)
    pull = (same   * dist).sum()                / same.sum().clamp(min=1)
    push = (diff_m * F.relu(margin - dist)).sum() / diff_m.sum().clamp(min=1)
    return pull + push


def velocity_loss(pred, target):
    """
    Temporal smoothness loss (NVIDIA A2F-3D Eq. 1: L_motion).
    Penalises the error between predicted velocity and ground-truth velocity
    across consecutive batch samples.

    Note: works best when the DataLoader preserves temporal ordering of clips
    (shuffle=False within a clip). With shuffle=True the loss still acts as a
    soft smoothness regulariser — consecutive batch items tend to have similar
    ground-truth velocities within the same emotion class.

    Shape: pred, target — (B, 260)
    Returns scalar.
    """
    if pred.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device)
    pred_vel = pred[1:]   - pred[:-1]                   # (B-1, 260)
    gt_vel   = target[1:] - target[:-1]                 # (B-1, 260)
    return F.mse_loss(pred_vel, gt_vel)


def vol_stab_loss(pred, audio, beta=5.0):
    """
    Volume-based stability regularisation (NVIDIA A2F-3D Eq. 2: L_vol_stab).
    During quiet audio segments the model should produce near-zero motion.
    High-weight when audio RMS is low → strongly penalise motion.
    Low-weight when audio RMS is high → let the model move freely.

    audio : (B, T_audio)  — raw waveform window at 16 kHz
    pred  : (B, 260)      — predicted blendshape frame
    beta  : controls steepness; higher = sharper quiet penalty
    """
    if pred.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device)
    rms      = audio.pow(2).mean(dim=-1).sqrt()          # (B,)
    w        = torch.exp(-beta * rms)                    # (B,)  high when quiet
    vel      = (pred[1:] - pred[:-1]).pow(2).mean(dim=-1)  # (B-1,)
    w_pairs  = (w[1:] + w[:-1]) / 2                     # (B-1,)
    return (w_pairs * vel).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"Losses → contra_w={args.contra_weight} | vel_w={args.vel_weight} "
          f"| vol_stab_w={args.vol_stab_weight}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, _, stats = build_dataloaders(
        args.tensor_dir, args.batch_size, args.num_workers,
        folder_mode=args.folder_mode)
    torch.save(stats, os.path.join(args.checkpoint_dir, 'dataset_stats_v3.pt'))

    model = EmpatheConciergeV3(num_emotions=5).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # ── Load pre-trained encoder weights (optional) ───────────────────────
    if args.encoder_weights and os.path.exists(args.encoder_weights):
        ckpt_enc = torch.load(args.encoder_weights, map_location=device, weights_only=False)
        model.audio_enc.load_state_dict(ckpt_enc['encoder_state_dict'])
        print(f"Loaded pre-trained encoder from: {args.encoder_weights}")
        print(f"  pre-trained val loss: {ckpt_enc.get('val_loss', 'n/a')}, epoch: {ckpt_enc.get('epoch', 'n/a')}")
    elif args.encoder_weights:
        print(f"[WARNING] --encoder_weights path not found: {args.encoder_weights}")

    start_epoch, best_val = 0, float('inf')
    history = {
        'train': [], 'val': [],
        'mse': [], 'contra': [],
        'vel': [], 'vol_stab': []
    }

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        best_val    = ckpt.get('best_val', float('inf'))
        # merge history keys safely (resume from V2 checkpoint won't have vel/vol_stab)
        saved_hist  = ckpt.get('history', {})
        for k in history:
            history[k] = saved_hist.get(k, [])
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.5f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (
        (ep + 1) / warmup if ep < warmup else
        0.5 * (1 + torch.cos(
            torch.tensor((ep - warmup) / max(1, args.epochs - warmup) * 3.14159)
        ).item())
    ))

    use_amp = device.type == 'cuda'
    scaler  = GradScaler() if use_amp else None
    amp_ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()

    header = f"{'Ep':>4} {'Train':>8} {'MSE':>8} {'Contra':>8} {'Vel':>8} {'VolStb':>8} {'Val':>8}"
    print(f"\n{header}")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_mse = ep_contra = ep_vel = ep_vol = ep_tot = 0.0

        for batch in train_loader:
            # dataset returns (audio, curves, emotion) or (audio, curves, emotion, intensity)
            # — both are supported; intensity defaults to None (=1.0) if absent
            audio, curves = batch[0].to(device), batch[1].to(device)
            emotion = batch[2].to(device).view(-1)
            intensity = batch[3].to(device).view(-1, 1).float() if len(batch) > 3 else None
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                # ── Core AWMSE loss (anatomically weighted, inside model) ──
                mse = model.training_loss(audio, curves, emotion, intensity)

                # ── Single differentiable denoiser step ────────────────────
                # Replaces the two separate sample() calls that existed before.
                # model.sample() is @torch.no_grad() — cannot backprop.
                # Instead: one forward pass through the denoiser at a random
                # timestep → reconstruct x0 via DDPM posterior mean formula.
                # All three regularisation losses use this pred_diff so the
                # gradient flows: loss → pred_diff → denoiser → encoder weights.
                t_diff      = torch.randint(0, model.T, (audio.shape[0],), device=device)
                noise_d     = torch.randn_like(curves)
                x_t_diff, _ = model.schedule.q_sample(curves, t_diff, noise_d)
                cond_diff   = model.encode_condition(audio, emotion, t_diff, intensity)
                eps_pred    = model.denoiser(x_t_diff, cond_diff)
                a_cp        = model.schedule.alphas_cp[t_diff][:, None]   # (B,1)
                pred_diff   = (x_t_diff - (1 - a_cp).sqrt() * eps_pred) / a_cp.sqrt()

                # ── All three regularisation losses use pred_diff ──────────
                # contra: emotions must produce geometrically distinct blendshapes
                # vel:    consecutive predictions should have smooth velocity
                # vol_stab: model should be still during silent audio segments
                contra   = contrastive_loss(pred_diff, emotion, args.contra_margin)
                vel      = velocity_loss(pred_diff, curves)
                vol_stab = vol_stab_loss(pred_diff, audio, beta=args.vol_stab_beta)

                loss = (mse
                        + args.contra_weight  * contra
                        + args.vel_weight     * vel
                        + args.vol_stab_weight * vol_stab)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ep_mse     += mse.item()
            ep_contra  += contra.item()
            ep_vel     += vel.item()
            ep_vol     += vol_stab.item()
            ep_tot     += loss.item()

        scheduler.step()
        n = len(train_loader)
        ep_mse    /= n; ep_contra /= n
        ep_vel    /= n; ep_vol    /= n; ep_tot /= n

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio, curves = batch[0].to(device), batch[1].to(device)
                emotion = batch[2].to(device).view(-1)
                intensity_v = batch[3].to(device).view(-1, 1).float() if len(batch) > 3 else None
                with amp_ctx:
                    val_loss += model.training_loss(audio, curves, emotion, intensity_v).item()
        val_loss /= len(val_loader)

        # ── History ──────────────────────────────────────────────────────
        history['train'].append(ep_tot)
        history['val'].append(val_loss)
        history['mse'].append(ep_mse)
        history['contra'].append(ep_contra)
        history['vel'].append(ep_vel)
        history['vol_stab'].append(ep_vol)

        star = ' ★' if val_loss < best_val else ''
        print(f"{epoch+1:4d} {ep_tot:8.5f} {ep_mse:8.5f} {ep_contra:8.5f} "
              f"{ep_vel:8.5f} {ep_vol:8.5f} {val_loss:8.5f}{star}")

        # ── Save best ────────────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {'model': model.state_dict(), 'epoch': epoch + 1,
                 'best_val': best_val, 'history': history},
                os.path.join(args.checkpoint_dir, 'best_v3.pt')
            )

        # ── Periodic checkpoint every 25 epochs ─────────────────────────
        if (epoch + 1) % 25 == 0:
            torch.save(
                {'model': model.state_dict(), 'epoch': epoch + 1,
                 'best_val': best_val, 'history': history},
                os.path.join(args.checkpoint_dir, f'epoch_{epoch+1:03d}_v3.pt')
            )

    torch.save(history, os.path.join(args.checkpoint_dir, 'history_v3.pt'))
    print(f"\n✓ Done. Best val: {best_val:.5f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('EmpatheConcierge V3 — Training Curves', fontweight='bold', fontsize=14)

        # Panel 1: Main loss curves
        ax = axes[0, 0]
        ax.plot(history['train'], label='Total',  color='steelblue')
        ax.plot(history['val'],   label='Val',    color='orange')
        ax.plot(history['mse'],   label='MSE',    color='green',  linestyle='--', alpha=0.7)
        ax.set(xlabel='Epoch', ylabel='Loss', title='Loss Curves')
        ax.legend(); ax.grid(alpha=0.3)

        # Panel 2: Contrastive loss
        ax = axes[0, 1]
        ax.plot(history['contra'], color='red', label='Contrastive')
        ax.set(xlabel='Epoch', ylabel='Contrastive Loss',
               title='Emotion Separation\n(↓ = emotions diverging)')
        ax.legend(); ax.grid(alpha=0.3)

        # Panel 3: Velocity loss
        ax = axes[1, 0]
        ax.plot(history['vel'], color='purple', label='Velocity')
        ax.set(xlabel='Epoch', ylabel='Velocity Loss',
               title='Temporal Smoothness\n(↓ = less jitter)')
        ax.legend(); ax.grid(alpha=0.3)

        # Panel 4: Vol-stab loss
        ax = axes[1, 1]
        ax.plot(history['vol_stab'], color='brown', label='Vol-Stab')
        ax.set(xlabel='Epoch', ylabel='Vol-Stab Loss',
               title='Silence Stability\n(↓ = quieter during pauses)')
        ax.legend(); ax.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(args.checkpoint_dir, 'training_curve_v3.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {out}")
    except Exception as e:
        print(f"[Plot skipped: {e}]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tensor_dir',       type=str,   required=True)
    p.add_argument('--epochs',           type=int,   default=150)
    p.add_argument('--batch_size',       type=int,   default=32)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--num_workers',      type=int,   default=2)
    p.add_argument('--checkpoint_dir',   type=str,   default='checkpoints_v3')
    p.add_argument('--resume',           type=str,   default=None)
    # Loss weights
    p.add_argument('--contra_weight',    type=float, default=0.05,
                   help='Weight for contrastive loss')
    p.add_argument('--contra_margin',    type=float, default=0.35,
                   help='Contrastive margin (raised from 0.30 for 5-class space)')
    p.add_argument('--vel_weight',       type=float, default=0.10,
                   help='Weight for velocity (temporal smoothness) loss')
    p.add_argument('--vol_stab_weight',  type=float, default=0.05,
                   help='Weight for volume-based stability loss')
    p.add_argument('--vol_stab_beta',    type=float, default=5.0,
                   help='Steepness of quiet-audio penalty in vol_stab_loss')
    p.add_argument('--encoder_weights',  type=str,   default=None,
                   help='Path to encoder_pretrained.pt from pretrain_encoder.py')
    p.add_argument('--folder_mode',      action='store_true',
                   help='Use Emotion_Train/Test subfolder structure (required for full dataset)')
    train(p.parse_args())
