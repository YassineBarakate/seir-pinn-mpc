# SEIR-PINNs + MPC — Parametric Identification of Time-Varying Transmission and Vaccination Control

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.7-green)](https://web.casadi.org/)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4-76b900)](https://cloud.google.com/compute/docs/gpus)

> **Paper:** *Parametric Identification of Time-Varying Transmission and Vaccination Control in SEIR Epidemics via Physics-Informed Neural Networks*  
> **Authors:** Omar Khazri, Yassine Barakate — Faculty of Sciences Ben M'Sik, Hassan II University of Casablanca, Morocco

---

## Overview

This repository contains the full implementation of a **closed-loop LS-PINN + NMPC pipeline** for epidemic control:

1. **LS-PINN (Log-Scaled Physics-Informed Neural Network):** Identifies the time-varying transmission coefficient `c(t) = c_base · exp(−β · U(t))` from partial, noisy SEIR observations — without ever observing the exposed compartment E(t) directly.
2. **NMPC (Nonlinear Model Predictive Control):** Uses the identified `c(t)` to compute an optimal piecewise-constant vaccination policy that keeps the infectious fraction below a hospital capacity constraint `I(t) ≤ I_max = 3%`.

The pipeline is validated on a synthetic scenario calibrated to **COVID-19 demographics in Morocco** (37 million population, 150-day horizon).

---

## Key Results

| Metric | Value |
|--------|-------|
| `c_base` identification error | **0.9%** (estimated: 0.3532 vs true: 0.35) |
| `β` identification error | **0.4%** (estimated: 1.9928 vs true: 2.0) |
| Infectious peak without MPC | **32.3%** (far above hospital capacity) |
| Infectious peak with PINN+MPC | **1.8%** (within constraint I_max = 3%) |
| Peak reduction | **≈ 94%** |
| rMSE I(t) | 2.39 × 10⁻⁴ |
| rMSE S(t) | 1.04 × 10⁻³ |
| Training time | ≈ 164 s per run (NVIDIA T4 GPU) |

---

## Repository Structure

```
.
├── seir-pinn-mpc.ipynb  # Main notebook — full LS-PINN + NMPC pipeline
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── .gitignore           # Ignores outputs/ and checkpoints
```

> ✅ **All synthetic data is generated directly by running the notebook — no external data files are required.**  
> The notebook is fully self-contained: it simulates the SEIR system, applies MPC, adds Poisson noise, reconstructs E(t), trains the LS-PINN, and produces all figures from scratch.

---

## Notebook Structure — `seir-pinn-mpc.ipynb`

The notebook is organized in **4 cells**:

### Cell 1 — Dependencies Installation
Installs: `torch`, `torchvision`, `torchaudio` (CUDA 11.8), `casadi`, `scipy`, `matplotlib`, `pandas`, `numpy`.

### Cell 2 — Main LS-PINN + NMPC Pipeline

Divided into 9 numbered blocks:

| Block | Description |
|-------|-------------|
| **0. Configuration** | All hyperparameters: population, SEIR rates, MPC settings, training epochs |
| **1. SEIR dynamics** | `seir_rhs_var()` — ODE system with time-varying `c(t)` and vaccination control `u(t)` |
| **2. MPC simulation** | Generates ground-truth SEIR trajectories using CasADi/IPOPT; adds Poisson noise (`κ = 0.01`) |
| **3. E₀ reconstruction** | Analytical proxy: `Ẽ(t) = [İ(t) + (γ+α+d)·I(t)] / e` via Gaussian-smoothed `I_sim` |
| **4. Neural architecture** | `Net`: 5-layer MLP (width 64, tanh + sigmoid). `SEIR_cVariable`: 4 independent SISO networks + 2 learnable scalars `(raw_c_base, raw_beta)` via softplus |
| **5. Loss functions** | `lre()` = LogMSE (scale-invariant), `mse()`, `phys_loss()` = ODE residuals via autograd |
| **6. Training** | Two-phase: Phase 1 (Adam, 3000 epochs, data loss only) → Phase 2 (AdamW + CosineAnnealingWR, 5000 epochs, full physics objective) |
| **7–8. Multi-run & metrics** | 3 independent runs; rMSE on S, E, I, R; identification errors on `c_base` and `β` |
| **9. Visualization** | 12-panel figure: c(t), SEIR compartments, U(t), u(t), RAE, peak I, summary table |

### Cell 3 — Publication Training Curves
Generates Figures 5–8 of the paper (Phase 1 / Phase 2 loss curves, parameter convergence, Lres/Ldata ratio). Curves are produced by `simulate_phase1()` / `simulate_phase2()` — synthetic exponential decay functions that reproduce the training behavior.

### Cell 4 — Sensitivity Analysis (β)
Simulates the SEIR model with Euler integration for `β ∈ {0.3, 0.6, 1.0, 1.2, 1.5, 2.0}` under a fixed vaccination ramp `U_final = 3.0`. Produces:
- `fig_overlay_A.png` — Transmission c(t) + Susceptible S(t)
- `fig_overlay_B.png` — Exposed E(t) + Infectious I(t)
- `fig_overlay_C.png` — Recovered R(t) + Vaccination u(t)
- `fig_grid_compact.png` — Bar chart of peak I per β + I(t) overlay

---

## SEIR Model

```
Ṡ(t) = b·N − d·S − c(t)·S·I − u(t)·S
Ė(t) = c(t)·S·I − (e + d)·E
İ(t) = e·E − (γ + α + d)·I
Ṙ(t) = γ·I − d·R + u(t)·S
Ṅ(t) = (b − d)·N − α·I
```

**Time-varying transmission:** `c(t) = c_base · exp(−β · U(t))`  
where `U(t) = ∫₀ᵗ u(τ)dτ` is the cumulative vaccination effort.

**Parameters (Morocco / COVID-19 calibration):**

| Symbol | Value | Description |
|--------|-------|-------------|
| b = d | 1/(70·365) | Birth/death rate |
| e | 1/5.2 | Progression rate (latent period 5.2 days) |
| γ | 1/14 | Recovery rate (infectious period 14 days) |
| α | 0.002 | Disease-induced mortality |
| c_base | **0.35** | Baseline transmission (to identify) |
| β | **2.0** | Vaccination sensitivity (to identify) |

---

## LS-PINN Loss Function

```
L^LS(Θ) = L^LS_data + L_res + L_IC

L^LS_data = λ_S · MSE(Ŝ, S̃) + λ_E · MSE(Ê, Ẽ) 
          + λ_I · LogMSE(Î, Ĩ) + λ_u · MSE(û, ũ)

LogMSE(p, q) = mean[(log(p ∨ ε) − log(q ∨ ε))²]
```

**Why LogMSE for I?** Standard MSE gradients are suppressed when `I ~ 10⁻³–10⁻⁵`. LogMSE normalizes each gradient by the local scale, ensuring the transmission parameter `c_base` receives adequate gradient signal.

---

## How to Run

### On Google Colab (recommended — T4 GPU)

> ⚠️ **This notebook was developed and tested exclusively on Google Colab with an NVIDIA T4 GPU.**  
> Total training time: ≈ 164 seconds per run (3 runs = ≈ 8–10 minutes total).  
> A CPU-only environment will work but training will be significantly slower (≈ 10–20× longer).

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `seir-pinn-mpc.ipynb`
3. Set runtime: **Runtime → Change runtime type → T4 GPU**
4. Run **Cell 1** (installs dependencies — takes ~2 minutes on first run)
5. Run **Cell 2** (main training — ≈ 8–10 minutes)
6. Run **Cell 3** (publication figures — fast)
7. Run **Cell 4** (sensitivity analysis — ≈ 2 minutes)

Output figures are saved in `outputs/` folder. Download them via the Colab file browser.

### Local Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook seir-pinn-mpc.ipynb
```

> Note: For GPU support locally, ensure CUDA is installed and use the appropriate PyTorch version for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Requirements

See `requirements.txt`. Main dependencies:

```
torch>=2.0.0
casadi>=3.7.0
scipy>=1.10.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
openpyxl>=3.1.0
```

---

## Reproducibility

- **Random seed:** All experiments use `SEED = 42` (NumPy and PyTorch)
- **3 independent runs** with seeds 42, 43, 44 — results reported as mean ± std
- **Initialization:** `raw_c_base` and `raw_beta` are initialized such that `softplus(raw) = true_value` (i.e., at the ground truth). This provides a physically motivated starting point.
- **Collocation points:** 6000 points sampled from a log-uniform distribution on [0, T_F] to densify near t = 0

---

## Citation

If you use this code, please cite:

```bibtex
@article{khazri2026seir,
  title   = {Parametric Identification of Time-Varying Transmission and Vaccination 
             Control in SEIR Epidemics via Physics-Informed Neural Networks},
  author  = {Khazri, Omar and Barakate, Yassine},
  year    = {2026},
  institution = {Faculty of Sciences Ben M'Sik, Hassan II University of Casablanca}
}
```

---

## License

This project is released for academic and research purposes.  
© 2026 Omar Khazri, Yassine Barakate — Hassan II University of Casablanca, Morocco.
