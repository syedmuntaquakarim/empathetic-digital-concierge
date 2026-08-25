"""
model_v2.py — EmpatheConcierge V2
==================================
PILLAR 2: Exact anatomical weighted MSE loss (from blendshape_map.py)
PILLAR 3: CNN + Bidirectional GRU AudioEncoder
Stronger StyleEncoder (256-dim), Larger DenoisingMLP (768-dim, 8 blocks)
4 emotions: Joy=0, Anger=1, Grief=2, Neutral=3
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from blendshape_map import LOSS_WEIGHTS


class TemporalAudioEncoder(nn.Module):
    """CNN frontend + BiGRU for temporal memory over 200ms audio window."""
    def __init__(self, out_dim=256, gru_hidden=256, gru_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,   64,  9, stride=4, padding=4), nn.GELU(), nn.BatchNorm1d(64),
            nn.Conv1d(64,  128, 5, stride=4, padding=2), nn.GELU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 5, stride=4, padding=2), nn.GELU(), nn.BatchNorm1d(256),
        )
        self.gru = nn.GRU(
            input_size=256, hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=0.1 if gru_layers > 1 else 0.0,
        )
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden * 2, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        feat = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        out, _ = self.gru(feat)
        return self.proj(out[:, -1, :])


class StyleEncoder(nn.Module):
    """
    Emotion + intensity conditioning.
    Accepts:
        e         : (B,)    long   — emotion class index
        intensity : (B, 1)  float  — scalar 0.0-1.0 (T1=0.3, T2=0.6, T3=0.9, T4=T5=1.0)
                                     Pass torch.ones(B,1) if intensity not available.
    """
    def __init__(self, num_emotions=4, out_dim=256):
        super().__init__()
        self.embed         = nn.Embedding(num_emotions, out_dim)
        self.intensity_proj = nn.Linear(1, out_dim)          # scalar → same space as embedding
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(), nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )

    def forward(self, e, intensity=None):
        emb = self.embed(e)                                   # (B, out_dim)
        if intensity is not None:
            emb = emb + self.intensity_proj(intensity)        # additive conditioning
        return self.mlp(emb)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args  = t[:, None].float() * freqs[None]
        return self.proj(torch.cat([args.sin(), args.cos()], dim=-1))


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    def forward(self, x): return x + self.net(x)


class DenoisingMLP(nn.Module):
    def __init__(self, curves_dim=260, audio_dim=256, style_dim=256, time_dim=128, hidden_dim=768, depth=8):
        super().__init__()
        cond_dim = audio_dim + style_dim + time_dim
        self.input_proj  = nn.Linear(curves_dim + cond_dim, hidden_dim)
        self.blocks      = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.film_layers = nn.ModuleList([nn.Linear(cond_dim, hidden_dim*2) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, curves_dim))
    def forward(self, x_t, cond):
        h = self.input_proj(torch.cat([x_t, cond], dim=-1))
        for block, film in zip(self.blocks, self.film_layers):
            gamma, beta = film(cond).chunk(2, dim=-1)
            h = block(h) * (1 + gamma) + beta
        return self.out(h)


class DDPMSchedule(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        self.register_buffer('betas',     betas)
        self.register_buffer('alphas',    alphas)
        self.register_buffer('alphas_cp', torch.cumprod(alphas, dim=0))
        self.T = T
    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        a = self.alphas_cp[t][:, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise


class EmpatheConciergeV2(nn.Module):
    EMOTION_MAP = {'joy': 0, 'anger': 1, 'grief': 2, 'neutral': 3}

    def __init__(self, curves_dim=260, num_emotions=4, T=1000):
        super().__init__()
        self.curves_dim = curves_dim
        self.T          = T
        self.audio_enc  = TemporalAudioEncoder(out_dim=256, gru_hidden=256, gru_layers=2)
        self.style_enc  = StyleEncoder(num_emotions=num_emotions, out_dim=256)
        self.time_emb   = TimestepEmbedder(dim=128)
        self.denoiser   = DenoisingMLP(curves_dim=curves_dim, audio_dim=256,
                                       style_dim=256, time_dim=128, hidden_dim=768, depth=8)
        self.schedule   = DDPMSchedule(T=T)
        self.register_buffer('loss_weights', LOSS_WEIGHTS.clone())

    def encode_condition(self, audio, emotion, t, intensity=None):
        """
        intensity: (B, 1) float tensor, 0.0-1.0. If None, defaults to 1.0 (full intensity).
        Backward compatible — all existing calls without intensity still work.
        """
        if intensity is None:
            intensity = torch.ones(audio.shape[0], 1, device=audio.device)
        return torch.cat([self.audio_enc(audio), self.style_enc(emotion, intensity), self.time_emb(t)], dim=-1)

    def training_loss(self, audio, curves, emotion, intensity=None):
        """intensity: (B,1) float 0-1. None = full intensity (backward compatible)."""
        B = audio.shape[0]
        t = torch.randint(0, self.T, (B,), device=audio.device)
        noise = torch.randn_like(curves)
        x_t, _ = self.schedule.q_sample(curves, t, noise)
        cond = self.encode_condition(audio, emotion, t, intensity)
        pred_noise = self.denoiser(x_t, cond)
        sq_err = (pred_noise - noise) ** 2
        return (sq_err * self.loss_weights).mean()

    @torch.no_grad()
    def sample(self, audio, emotion, T_steps=200, intensity=None):
        B, device = audio.shape[0], audio.device
        sched = self.schedule
        indices = torch.linspace(0, self.T-1, T_steps).long().tolist() if T_steps < self.T else list(range(self.T-1, -1, -1))
        x = torch.randn(B, self.curves_dim, device=device)
        for i in reversed(indices):
            t_b  = torch.full((B,), i, device=device, dtype=torch.long)
            cond = self.encode_condition(audio, emotion, t_b, intensity)
            eps  = self.denoiser(x, cond)
            alpha, a_cp, beta = sched.alphas[i], sched.alphas_cp[i], sched.betas[i]
            mean = (1.0/alpha.sqrt()) * (x - beta/(1-a_cp).sqrt() * eps)
            x    = mean + beta.sqrt() * torch.randn_like(x) if i > 0 else mean
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EmpatheConciergeV3(EmpatheConciergeV2):
    """
    V3 — drop-in upgrade of V2 with 5 emotions (adds Amazement).
    All architecture is identical; only EMOTION_MAP and default
    num_emotions change. train_v3.py imports this class.

    Emotion indices:
        Joy=0  Anger=1  Grief=2  Neutral=3  Amazement=4
    """
    EMOTION_MAP = {'joy': 0, 'anger': 1, 'grief': 2, 'neutral': 3, 'amazement': 4}

    def __init__(self, curves_dim=260, num_emotions=5, T=1000):
        super().__init__(curves_dim=curves_dim, num_emotions=num_emotions, T=T)


if __name__ == '__main__':
    print("=" * 55)
    print("── V2 smoke test (4 emotions) ──")
    model = EmpatheConciergeV2()
    model.eval()
    audio   = torch.randn(4, 3200)
    curves  = torch.randn(4, 260)
    emotion = torch.randint(0, 4, (4,))
    loss = model.training_loss(audio, curves, emotion)
    print(f"Loss: {loss.item():.4f}  |  Params: {model.count_parameters():,}")
    w = model.loss_weights
    print(f"browDownL  dim0  ANGER: {w[0].item():.1f}")
    print(f"browRaiseInL dim4 GRIEF: {w[4].item():.1f}")
    print(f"mouthCornerPullL dim77 JOY: {w[77].item():.1f}")
    print(f"HeadYaw    dim251 HEAD : {w[251].item():.2f}")
    print("V2 smoke test passed ✓")

    print("\n── V3 smoke test (5 emotions, includes Amazement) ──")
    model3   = EmpatheConciergeV3()
    model3.eval()
    emotion5 = torch.randint(0, 5, (4,))
    loss3    = model3.training_loss(audio, curves, emotion5)
    print(f"Loss: {loss3.item():.4f}  |  Params: {model3.count_parameters():,}")
    pred3    = model3.sample(audio, emotion5, T_steps=10)
    print(f"Sample shape: {pred3.shape}  (expected: [4, 260])")
    print("V3 smoke test passed ✓")
