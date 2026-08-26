# Empathetic Digital Concierge: Audio-Driven 3D Facial Animation via Style-Conditioned Diffusion

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Unreal Engine 5](https://img.shields.io/badge/Unreal_Engine-5.3+-black.svg)](https://www.unrealengine.com/)

> **Academic Major Project** | Department of Computer Engineering, Jamia Millia Islamia (2025–2026)  
> **Authors:** Syed Muntaqua Karim & Misbah Akhtar  
> **Supervisor:** Prof. Dr. Sarfaraz Masood  

📄 **[Read the full Technical Paper here](Empathetic_Digital_Concierge_Paper.pdf)**

An end-to-end **Denoising Diffusion Probabilistic Model (DDPM)** that synthesizes emotionally expressive, 260-dimensional ARKit blendshape sequences for Unreal Engine 5 MetaHumans directly from 16 kHz monaural speech. 

Unlike traditional deterministic regressors (LSTMs, Transformers) that suffer from *magnitude collapse* (averaging expressions to a neutral state), or standard VAEs that suffer from *posterior collapse* on sparse blendshape data, this architecture utilizes **Feature-wise Linear Modulation (FiLM)** and an **Anatomically Weighted Mean Squared Error (AWMSE)** to preserve identity-specific micro-expressions and sharp emotional fidelity.
---

## 📌 Key Architectural Contributions

1. **FiLM-Conditioned Denoising:** Emotion conditioning is injected at *every* residual block via FiLM ($h' = \gamma \cdot h + \beta$) rather than simple input concatenation, fundamentally reshaping the denoiser's computation.
2. **Continuous Intensity Interpolation:** Emotion is parameterized by a discrete class label and a continuous scalar $\alpha \in [0, 1]$, enabling fluid interpolation between mild (30%) and maximal (110%) expression amplitudes without retraining.
3. **Anatomically Weighted MSE (AWMSE):** FACS-derived loss weighting amplifies gradient signals on anchor muscles (e.g., 9.0x on Frontalis/Amazement, 8.0x on Zygomaticus/Joy) while suppressing massive head pose rotations (0.05x) to prevent them from dominating the loss.
4. **Volume-Conditioned Stability Loss:** A temporal regularizer weighted by $\exp(-\beta \cdot \text{RMS}(\text{audio}))$ strictly suppresses facial hallucination and jitter during silent speech intervals.

---

## 🏗️ System Pipeline

```
16kHz Audio ──> [CNN-BiGRU Encoder] ──────┐
                                          v
Emotion + Intensity ──> [Style MLP] ──> [FiLM Modulation] ──> [8-Block ResNet DDPM] ──> 260-dim ARKit CSV
```

### Dataset & Synchronization
Trained on a custom FACS-compliant dataset of 375 recordings across 5 emotions (Joy, Anger, Grief, Amazement, Neutral) at 5 intensity tiers. Audio and facial motion capture were autonomously synchronized via a custom multi-threaded ETL pipeline using a **rolling 50ms kinetic-energy transient detection algorithm** to align physical clap impulses.

---

## 📊 Quantitative Benchmarks

Evaluation on 115 strictly held-out test clips. The DDPM successfully overcomes the amplitude-attenuated regression ceiling (~48%) where deterministic baselines fail.

| Architecture | Mean Cosine Similarity (CS) ↑ | Notes |
| :--- | :--- | :--- |
| **Our DDPM (FiLM + AWMSE)** | **79.14%** | **Strong angular fidelity; no magnitude collapse.** |
| LSTM Regressor | 48.79% | Amplitude-attenuation (mean face regression). |
| 1D-ResNet Regressor | 47.03% | Lacks temporal context processing. |
| Transformer Regressor | 48.47% | Deterministic decoding bottleneck. |
| Conditional VAE | 24.40% | Severe posterior collapse on sparse 260-dim space. |

*Per-Emotion CS (Ours): Anger (84.00%), Grief (82.40%), Joy (79.60%), Amazement (78.80%), Neutral (72.30%).*

---

## 🚀 Quickstart & UE5 Integration

### 1. Training & Inference
```bash
git clone [https://github.com/syedmuntaquakarim/empathetic-digital-concierge.git](https://github.com/syedmuntaquakarim/empathetic-digital-concierge.git)
cd empathetic-digital-concierge

# Run inference on a raw .wav file
python src/inference_v3.py --audio_path input.wav --emotion joy --intensity 0.9
```

### 2. Unreal Engine 5 MetaHuman Deployment
The generated `.csv` output is mapped directly to MetaHuman control rigs. Use the included Python-UE5 controller to bake the curves natively:
```bash
# Execute within the Unreal Engine Python Environment
python ue5_integration/bake_to_anim_sequence.py --input curves.csv --target_skeleton /Game/MetaHumans/Face/Face_Archetype_Skeleton
```

---
## 📜 Research & Citations
- Karim, S.M. & Akhtar, M. *"The Empathetic Digital Concierge: Emotionally Expressive Audio-Driven 3D Facial Animation via Style-Conditioned Diffusion"*, Jamia Millia Islamia, 2026.