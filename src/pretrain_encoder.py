"""
pretrain_encoder.py — TemporalAudioEncoder Pre-Training
=========================================================
Trains ONLY the CNN+BiGRU audio encoder on neutral phonetic clips
(your 75 recorded videos) to establish a strong phonetic baseline
before DDPM emotion training.

Task:    audio window (3200 samples) → 260-dim blendshape frame
Loss:    Lip-weighted MSE  (emphasises mouth/jaw, suppresses brows/head)
Output:  encoder_pretrained.pt  (encoder weights only, ready to load into V3)

Usage:
    python pretrain_encoder.py --tensor_dir "D:\\Dataset\\Neutral_Tensors"

After this completes, run train_v3.py with:
    --encoder_weights "checkpoints_encoder\\encoder_pretrained.pt"
"""

import os, argparse, contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from model_v2   import TemporalAudioEncoder
from dataset_v2 import build_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Lip-region weights for pre-training
# ─────────────────────────────────────────────────────────────────────────────
# Pre-training goal is phonetic accuracy, not emotion discrimination.
# We heavily weight the lip/jaw articulation dims and suppress everything else.
# Brow dims are set to 0.1x — we don't want the encoder to "learn" brows
# from neutral data (brows are style/emotion, not phoneme-driven).

def build_lip_weights(curves_dim=260, device='cpu'):
    """
    Returns a (260,) weight tensor for lip-focused pre-training.

    Region               Dims        Weight    Reason
    ─────────────────── ─────────── ───────── ─────────────────────────────
    Jaw open             191         10.0x     Primary viseme signal
    Lip corners pull     77, 78      8.0x      EE / wide vowels
    Upper lip raise      79, 80      7.0x      F, V phonemes
    Lower lip depress    85, 86      7.0x      Open vowels
    Lip stretch          87, 88      7.0x      EE coarticulation
    Lip round            89, 90      8.0x      OO / W / rounded vowels
    Lip press            91, 92      9.0x      M, B, P bilabials (closure)
    Lip part             93          9.0x      Start of open vowels
    Jaw clench           196, 197    3.0x      Soft — not emotion-critical yet
    Brow dims            0-9         0.1x      Suppress — emotion, not phoneme
    Head pose            251-260     0.05x     Suppress — same as V3
    All others           —           1.0x      Neutral
    """
    w = torch.ones(curves_dim, device=device)

    # Suppress brows (emotion signal, not phoneme)
    w[0:10]    = 0.1

    # Suppress head pose
    w[251:260] = 0.05

    # Lip articulation — boosted
    w[77]  = 8.0   # mouthCornerPullL
    w[78]  = 8.0   # mouthCornerPullR
    w[79]  = 7.0   # upperLipRaiseL
    w[80]  = 7.0   # upperLipRaiseR
    w[85]  = 7.0   # lowerLipDepressL
    w[86]  = 7.0   # lowerLipDepressR
    w[87]  = 7.0   # lipStretchL
    w[88]  = 7.0   # lipStretchR
    w[89]  = 8.0   # lipRoundL   (OO)
    w[90]  = 8.0   # lipRoundR
    w[91]  = 9.0   # lipPressL   (M/B/P closure)
    w[92]  = 9.0   # lipPressR
    w[93]  = 9.0   # lipPart

    # Jaw
    w[191] = 10.0  # jawOpen     (primary viseme driver)
    w[196] = 3.0   # jawClenchL  (soft — not the focus here)
    w[197] = 3.0   # jawClenchR

    return w


# ─────────────────────────────────────────────────────────────────────────────
# Regression head (discarded after pre-training)
# ─────────────────────────────────────────────────────────────────────────────

class EncoderWithHead(nn.Module):
    """
    Thin wrapper: TemporalAudioEncoder + a small MLP regression head.
    The head maps encoder output (256-dim) → blendshapes (260-dim).
    After pre-training, only the encoder weights are saved.
    The head is thrown away — it was just a training scaffold.
    """
    def __init__(self, curves_dim=260, encoder_out_dim=256):
        super().__init__()
        self.encoder = TemporalAudioEncoder(
            out_dim=encoder_out_dim,
            gru_hidden=256,
            gru_layers=2
        )
        self.head = nn.Sequential(
            nn.Linear(encoder_out_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, curves_dim)
        )

    def forward(self, audio):
        return self.head(self.encoder(audio))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encoder_parameters(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # DataLoader — emotion labels exist in tensors but are ignored here
    train_loader, val_loader, _, stats = build_dataloaders(
        args.tensor_dir, args.batch_size, args.num_workers
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = EncoderWithHead(curves_dim=260, encoder_out_dim=256).to(device)
    print(f"Total params:   {model.count_parameters():,}")
    print(f"Encoder params: {model.encoder_parameters():,}  (these get saved)")

    # Lip-weighted loss
    lip_weights = build_lip_weights(device=device)

    def weighted_mse(pred, target):
        return ((pred - target).pow(2) * lip_weights).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # LR schedule: warmup then cosine decay
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

    best_val = float('inf')
    history  = {'train': [], 'val': []}

    print(f"\n{'Ep':>4} {'Train':>10} {'Val':>10} {'LR':>10}")
    print("=" * 45)

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0

        for audio, curves, _emotion in train_loader:
            # _emotion is intentionally unused — pre-training is emotion-agnostic
            audio, curves = audio.to(device), curves.to(device)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                pred = model(audio)
                loss = weighted_mse(pred, curves)

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

            ep_loss += loss.item()

        scheduler.step()
        ep_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for audio, curves, _emotion in val_loader:
                audio, curves = audio.to(device), curves.to(device)
                with amp_ctx:
                    val_loss += weighted_mse(model(audio), curves).item()
        val_loss /= len(val_loader)

        history['train'].append(ep_loss)
        history['val'].append(val_loss)

        current_lr = scheduler.get_last_lr()[0]
        star = ' ★' if val_loss < best_val else ''
        print(f"{epoch+1:4d} {ep_loss:10.5f} {val_loss:10.5f} {current_lr:10.2e}{star}")

        if val_loss < best_val:
            best_val = val_loss

            # ── Save ONLY the encoder weights ──────────────────────────────
            # This is the key output: just encoder.state_dict()
            # The head is discarded intentionally.
            encoder_path = os.path.join(args.checkpoint_dir, 'encoder_pretrained.pt')
            torch.save({
                'encoder_state_dict': model.encoder.state_dict(),
                'epoch':              epoch + 1,
                'val_loss':           best_val,
                'history':            history,
                'note': (
                    'Pre-trained TemporalAudioEncoder on neutral phonetic clips. '
                    'Load into EmpatheConciergeV3 via model.audio_enc.load_state_dict().'
                )
            }, encoder_path)

        # Full model checkpoint every 20 epochs (for resuming)
        if (epoch + 1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch':            epoch + 1,
                'best_val':         best_val,
                'history':          history,
            }, os.path.join(args.checkpoint_dir, f'encoder_epoch_{epoch+1:03d}.pt'))

    print(f"\n✓ Pre-training complete. Best val: {best_val:.5f}")
    print(f"✓ Encoder weights saved → {args.checkpoint_dir}/encoder_pretrained.pt")
    print(f"\nNext step — load into V3 training:")
    print(f"  python train_v3.py \\")
    print(f"    --tensor_dir <emotion_tensors> \\")
    print(f"    --encoder_weights {args.checkpoint_dir}/encoder_pretrained.pt \\")
    print(f"    --freeze_encoder_epochs 10")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle('TemporalAudioEncoder Pre-Training — Neutral Phonetic Clips',
                     fontweight='bold')
        ax.plot(history['train'], label='Train Loss', color='steelblue')
        ax.plot(history['val'],   label='Val Loss',   color='orange')
        ax.axhline(best_val, color='green', linestyle='--', alpha=0.6,
                   label=f'Best Val: {best_val:.5f}')
        ax.set(xlabel='Epoch', ylabel='Lip-Weighted MSE', title='Encoder Pre-Training Loss')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(args.checkpoint_dir, 'encoder_pretrain_curve.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {out}")
    except Exception as e:
        print(f"[Plot skipped: {e}]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Pre-train TemporalAudioEncoder on neutral phonetic clips'
    )
    p.add_argument('--tensor_dir',     type=str,   required=True,
                   help='Path to .pt tensor files from neutral recordings')
    p.add_argument('--epochs',         type=int,   default=80,
                   help='80 epochs sufficient for 75 clips; increase if val still falling')
    p.add_argument('--batch_size',     type=int,   default=16,
                   help='Smaller batch (16) better for small dataset')
    p.add_argument('--lr',             type=float, default=5e-4,
                   help='Higher LR than V3 — encoder-only, smaller model')
    p.add_argument('--num_workers',    type=int,   default=2)
    p.add_argument('--checkpoint_dir', type=str,   default='checkpoints_encoder')
    pretrain(p.parse_args())
