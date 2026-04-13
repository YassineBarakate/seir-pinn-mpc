# SEIR-PINNs + NMPC — COVID-19 Morocco
## LS-PINN Global Calibration with Optimal Vaccination Control

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![CasADi](https://img.shields.io/badge/CasADi-3.7-green)](https://web.casadi.org/)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20T4-76b900)](https://cloud.google.com/compute/docs/gpus)

> **Paper:** *Parametric Identification of Time-Varying Transmission and Vaccination Control in SEIR Epidemics via Physics-Informed Neural Networks*
> **Authors:** Omar Khazri, Yassine Barakate — Faculty of Sciences Ben M'Sik, Hassan II University of Casablanca, Morocco

---

## Overview

This repository contains the full implementation of a **closed-loop LS-PINN + NMPC pipeline** for epidemic control, calibrated on **real COVID-19 surveillance data from Morocco** (37 million population, 180-day horizon starting 2020-12-29).

The pipeline consists of three stages:

1. **Data preprocessing.** Raw OWID vaccination data and Morocco case data are smoothed, and the unobserved exposed compartment E(t) is analytically reconstructed from the infectious signal I(t).

2. **LS-PINN calibration.** A Physics-Informed Neural Network with scale-adapted loss functions (LogMSE on I, LRE on E) identifies the time-varying transmission rate `c(t) = nn_c(t, U(t))` and all SEIR compartments from partial, noisy observations.

3. **NMPC proof-of-concept.** The calibrated `c(t)` is used offline as a fixed exogenous predictor inside a Nonlinear Model Predictive Controller that computes an optimal vaccination policy keeping `I(t) ≤ I_max`.

---

## Key Results

| Metric | LS-PINN | PINN-std |
|--------|---------|----------|
| rMSE S(t) | 9.08e-10 | 8.69e-10 |
| rMSE E(t) | 6.85e-04 | 2.76e-01 |
| rMSE I(t) | 1.87e-04 | 5.04e-03 |
| rMSE R(t) | 4.94e-05 | 6.09e-05 |
| rMSE c(t) | 1.93e-05 | 2.97e-05 |
| **MAE_rel I(t)** | **0.94%** | **6.78%** |
| Training time | ≈ 193 s/run | ≈ 192 s/run |
| Epochs (Phase 1 + Phase 2) | 3 000 + 4 000 | 3 000 + 4 000 |

**NMPC result (proof of concept):**
Under the calibrated c(t), the NMPC brings I(t) below the threshold I_max = 11 099 individuals from day ~45 onward, using a piecewise vaccination policy of ~2 100–2 300 doses/day — compared to an uncontrolled baseline (u = 0) where I(t) remains persistently above the threshold.

---

## Key Differences vs. Prior Version

| Aspect | Previous version | **This version** |
|--------|-----------------|------------------|
| Data source | Synthetic (Poisson-noisy SEIR) | **Real Morocco COVID-19 data (OWID + HCP)** |
| c(t) model | Parametric `c_base·exp(−β·U(t))` | **Neural network `nn_c(t, U(t))`** |
| Loss on E | MSE (both models) | **LRE for LS-PINN, MSE×50 for PINN-std** |
| Loss on I | LogMSE (LS), MSE×100 (std) | Same — confirmed correct |
| EP_PHYS | 5 000 or 8 000 | **4 000** (loss plateau confirmed after ~ep 4000) |
| Architecture width | 64 | **32** |
| Collocation points | 6 000 | **8 000** |
| NMPC baseline | — | **u = 0 (uncontrolled)** |
| NMPC c(t) source | — | **LS-PINN offline calibration** |
| Comparison | Single model | **LS-PINN vs PINN-std** |

---

## Repository Structure

```
.
├── seir_pinn.ipynb          # Main notebook — 4 cells
├── maroc_covid_data.xls        # Morocco COVID-19 case data (required)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── outputs/
│   └── figures/
│       ├── figA1_lspinn_c.png        # LS-PINN c(t)
│       ├── figA2_lspinn_SEIR.png     # LS-PINN S,E,I,R
│       ├── figB_training_curves.png  # Real training curves
│       ├── figC1_comparison_c.png    # LS-PINN vs PINN-std c(t)
│       ├── figC2_comparison_SEIR.png # LS-PINN vs PINN-std S,E,I,R
│       └── figD_nmpc_comparison.png  # NMPC vs uncontrolled
└── .gitignore
```

> ⚠️ **External data required.**
> `maroc_covid_data.xls` (Morocco COVID-19 case counts) must be placed in the working directory before running the notebook.
> Vaccination data is downloaded automatically from OWID at runtime.

---

## Data Sources

| Data | Source | Period |
|------|--------|--------|
| COVID-19 cases (Active, Recovered, Deaths, Confirmed) | Morocco HCP / JHU CSSE | 2020-12-29 → +180 days |
| Vaccination (total_vaccinations) | Our World in Data (OWID) | Loaded at runtime from GitHub |

### E(t) Reconstruction

The exposed compartment E(t) is never directly observed. It is analytically reconstructed from the smoothed infectious signal:

```
Ẽ(t) = [dĨ/dt + (γ + α + d) · Ĩ(t)] / e
```

The same signal is used for both the supervision loss and the rMSE evaluation — a **single-signal strategy** that eliminates evaluation bias.

### c(t) Reconstruction

The transmission rate is reconstructed from E(t) and the SEIR equations:

```
c(t) = [dE/dt + (e + d) · E(t)] / [S(t) · I(t) + ε]
```

Applied with double Gaussian smoothing (σ = 5, then σ = 3) to remove numerical noise from differentiation.

---

## SEIR Model

```
Ṡ(t) = b·N − d·S − c(t)·S·I − u(t)·S
Ė(t) = c(t)·S·I − (e + d)·E
İ(t) = e·E − (γ + α + d)·I
Ṙ(t) = γ·I − d·R + u(t)·S
```

Conservation law enforced exactly in the network:

```
R̂ = clamp(1 − Ŝ − Ê − Î, 0, 1)
```

---

## Epidemiological Parameters

| Symbol | Value | Description | Source |
|--------|-------|-------------|--------|
| N_pop | 37 × 10⁶ | Morocco total population | HCP Morocco (2020) |
| b = d | 1/(70×365) day⁻¹ | Natural birth and death rates (stable population) | World Bank (2020) |
| e | 1/5.2 day⁻¹ | Latency rate (incubation period = 5.2 days) | Li et al. (2020), NEJM |
| γ | 1/14 day⁻¹ | Recovery rate (mean infectious period = 14 days) | WHO/CDC COVID-19 guidelines |
| α | 2×10⁻³ day⁻¹ | COVID-19 disease-induced mortality rate | Morocco IFR estimates |
| T_F | 180 days | Calibration horizon | — |
| VAX_START | Day 30 | Vaccination start day | OWID Morocco data |
| START_DATE | 2020-12-29 | Time series origin | JHU CSSE repository |

---

## Neural Network Architecture

| Component | Description |
|-----------|-------------|
| **FNet1D** | 1D Fourier-feature MLP — used for S(t), E(t), I(t), u(t) |
| **FNet2D** | 2D Fourier-feature MLP — used for c(t, U(t)) |
| Hidden layers | 5 layers, width 32, tanh activation |
| Output activation | Sigmoid (all networks) |
| Fourier features | N = 16 frequencies, scale = 2.0 (FNet1D) / 1.0 (U branch of FNet2D) |
| Weight initialization | Xavier normal |
| Conservation law | R = clamp(1 − S − E − I, 0, 1) — exact, not learned |

**Fourier embedding:**

```
φ(x) = [sin(Bx), cos(Bx)]   B ~ N(0, scale²)
```

This enables the networks to capture multi-frequency temporal patterns in c(t) without requiring explicit frequency specification.

---

## Loss Functions

### LS-PINN vs PINN-std — per compartment

| Compartment | LS-PINN loss | LS weight | PINN-std loss | STD weight |
|-------------|-------------|-----------|---------------|------------|
| S | MSE | 5.0 | MSE | 1.0 |
| **E** | **LRE** | 3.0 | MSE | 50.0 |
| **I** | **LogMSE** | 0.5 | MSE | 100.0 |
| u | MSE | 0.1 | MSE | 0.1 |
| c | MSE | 10.0 | MSE | 10.0 |
| IC (S₀, E₀, I₀) | MSE | 20 / 1 / 1 | MSE | 20 / 1 / 1 |

### Loss function definitions

```python
# LogMSE — used on I in LS-PINN
# Scale-invariant: normalizes gradient by local trajectory amplitude
LogMSE(p, q) = mean[ (log(p ∨ ε) − log(q ∨ ε))² ]        ε = 1e-9

# LRE (Log-Relative Error) — used on E in LS-PINN
# Penalizes relative deviation: self-normalizing, no heavy boost needed
LRE(p, q)    = mean[ ((p − q) / (|q| + ε))² ]

# MSE — used for S, R, u, c, IC, and all PINN-std compartments
MSE(p, q)    = mean[ (p − q)² ]
```

**Why LogMSE for I?**
I(t) oscillates in [7×10⁻⁵, 6.6×10⁻⁴]. Standard MSE gradients are suppressed by O(I²), making it impossible for the optimizer to fit small-amplitude infectious dynamics. LogMSE normalizes each gradient by the local scale, ensuring a consistent signal regardless of absolute magnitude.

**Why LRE for E?**
E(t) oscillates in [1.3×10⁻⁵, 2.3×10⁻⁴] — same scale issue. LRE provides relative penalization and eliminates the need for large boosting weights, which would otherwise destabilize the NTK adaptive weighting.

### Total loss — Phase 2

```
L_total = w_S·L_S + w_I·L_I + w_E·L_E + w_u·L_u + w_c·L_c    [L_data]
        + w_rS·rS + w_rE·rE + w_rI·rI + w_rR·rR               [L_phys]
        + w_IC·L_IC + L_term                                    [L_IC + terminal]
```

Physics residual weights (Phase 2 only, before NTK adaptation):

| Term | Initial weight |
|------|---------------|
| w_rS | 5 |
| w_rE | 20 |
| w_rI | 20 |
| w_rR | 15 |
| w_IC | 50 |

---

## Training Hyperparameters

| Hyperparameter | Symbol | Value |
|----------------|--------|-------|
| Collocation points | N_col | 8 000 |
| Collocation sampling | — | log-uniform on [0, T_F] — densifies near t = 0 |
| Independent runs | M | 3 (seeds 42, 43, 44) |
| Numerical floor | ε | 10⁻⁹ |
| **Phase 1 — Data fit** | | |
| Optimizer | — | Adam |
| Learning rate | η₁ | 10⁻³ |
| Scheduler | — | StepLR (γ = 0.5, step = 1 000) |
| Epochs | E₁ | 3 000 |
| **Phase 2 — Data + Physics + IC** | | |
| Optimizer | — | AdamW |
| Learning rate (S, E, I, u networks) | η₂ | 3×10⁻⁴ |
| Learning rate (c network) | η₂ᶜ | 8×10⁻⁴ |
| Scheduler | — | CosineAnnealingWarmRestarts (T₀ = 2 000, T_mult = 2) |
| Epochs | E₂ | **4 000** (plateau confirmed after ~ep 4 000) |
| Gradient clip norm | — | 1.0 |
| Weight decay | — | 10⁻³ |

---

## NTK Adaptive Weighting

During Phase 2, loss weights are automatically adapted every 200 epochs using a Neural Tangent Kernel-inspired strategy:

```
Step 1 — Compute gradient norm per loss term:
    gk = ‖∇_θ Lk‖₂

Step 2 — Compute inverse-norm weights (normalized):
    wk_raw = max_norm / (gk + ε)
    wk_norm = wk_raw · n / Σ wk_raw

Step 3 — Exponential moving average (momentum = 0.9):
    wk ← 0.9 · wk_prev + 0.1 · wk_norm
```

This prevents any single loss term from dominating the optimization landscape, particularly when gradient magnitudes differ by orders of magnitude across compartments (e.g., L_S vs L_I).

**Observed NTK behavior (LS-PINN):**
- w_S decreases from 5.0 → ~1.2 (S is easy to fit — gradient redistributed)
- w_I remains stable ~0.5 (LogMSE already self-normalizes)
- w_E decreases from 3.0 → ~0.8 (LRE convergence faster than MSE-based)

---

## NMPC — Proof of Concept

### Design philosophy

The NMPC is presented as a **proof of concept with offline-calibrated predictor**. The pipeline is intentionally decoupled:

```
LS-PINN calibration (offline)
         ↓
    c(t) fixed — not updated online
         ↓
NMPC uses c(t) as exogenous input
         ↓
Optimizes u*(t) over sliding horizon
```

The controller does **not** re-identify c(t) at each step. This design choice isolates the identification contribution of the LS-PINN from the control contribution of the NMPC, consistent with a proof-of-concept framing rather than a fully adaptive closed-loop system.

### NMPC Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Prediction horizon | N | 21 days | Rolling optimization window |
| Time step | Δt | 1 day | Euler integration step |
| Min vaccination rate | u_min | 0.0 | No forced vaccination |
| Max vaccination rate | u_max | 0.015 | ≈ 555 000 doses/day |
| Infectious threshold | I_max | 0.0003 | ≈ 11 099 individuals |
| Infection penalty | Q_I | 10⁶ | Penalizes I(t) > I_max |
| Effort penalty | R_U | 1.0 | Penalizes vaccination effort |
| Ramp-rate constraint | Δu_max | 0.001/day | Max daily change in u (smoothness) |
| Control start | — | Day 30 | Coincides with vaccination start |
| Control end | — | Day 180 | End of calibration horizon |
| Solver | — | IPOPT via CasADi | Nonlinear programming |
| Warm-start | — | Sliding: u*_{k+1} → init of next step | Accelerates convergence |

### Prediction model inside NMPC

At each time step t, the NMPC solves:

```
min_{u_0,...,u_{N-1}}  Σ_{k=0}^{N-1} [ Q_I · max(0, I_k − I_max)²  +  R_U · u_k² ]

subject to:
    SEIR Euler dynamics with c_k = c_PINN(t+k)   [fixed offline]
    u_min ≤ u_k ≤ u_max                           [box constraints]
    |u_k − u_{k-1}| ≤ Δu_max                      [ramp-rate constraints]

Apply u*_0 only → advance one day → repeat
```

### NMPC Result Interpretation

The NMPC identifies a front-loaded vaccination strategy (~300 000 doses/day in the first week, then ~2 100 doses/day in steady state) that brings I(t) below I_max from day ~45 onward. The uncontrolled baseline (u = 0) maintains I(t) persistently above the threshold throughout the horizon.

> **Baseline note:** The comparison is against u = 0 (no vaccination), not the historical OWID data. This is consistent with the proof-of-concept framing: the NMPC demonstrates the value of optimal control relative to the absence of any intervention.

---

## Notebook Structure

The notebook `seir_pinn.ipynb` is organized in **4 cells**:

### Cell 2 — Configuration, Data, Architecture, Loss Functions

| Block | Content |
|-------|---------|
| 0. Configuration | All hyperparameters: N_pop, SEIR rates, EP_DATA, EP_PHYS, loss weights, NTK settings |
| 1. Data loading | Morocco COVID-19 CSV + OWID vaccination data; Gaussian smoothing |
| 2. E(t) reconstruction | Single-signal strategy: Ẽ(t) from dĨ/dt |
| 3. c(t) reconstruction | Double Gaussian-smoothed reconstruction |
| 4. Architecture | FourierEmbedding, FNet1D, FNet2D, SEIRModel |
| 5. Loss functions | mse(), log_mse(), lre(), loss_I(), loss_E(), physics_residuals() |
| 6. NTK | gradient_norm(), ntk_weight_update() |
| 7. Metrics | rMSE(), mae_rel() |

### Cell 3 — Training, Metrics, Figures A/B/C

| Block | Content |
|-------|---------|
| train() | Phase 1 (Adam, data fit) → Phase 2 (AdamW, data + physics + IC) with full history logging |
| evaluate() | Inference on full time grid → numpy arrays |
| Experiment loop | M = 3 runs × 2 models → mean ± std results |
| Final metrics | rMSE(S,E,I,R,c), MAE_rel(I) printed |
| Figure A1 | LS-PINN c(t) — in-sample calibration |
| Figure A2 | LS-PINN S, E, I, R — in-sample calibration |
| Figure B | Real training curves: total loss · loss decomposition · NTK weights |
| Figure C1 | LS-PINN vs PINN-std — c(t) comparison |
| Figure C2 | LS-PINN vs PINN-std — S, E, I, R comparison |

### Cell 4 — NMPC and Figure D

| Block | Content |
|-------|---------|
| NMPC config | NMPC_CFG dict — all parameters in one place |
| seir_step() | Euler integration step for prediction and simulation |
| get_c() | Returns PINN-predicted c(t) at day t (frozen offline) |
| run_free() | Uncontrolled trajectory with u = 0 (baseline) |
| run_nmpc() | NMPC closed-loop with CasADi/IPOPT |
| Figure D | 2×3 grid: I(t), u*(t), S(t), E(t), R(t), U(t) cumulative — NMPC vs uncontrolled, t = 0 to 180 |

---

## Produced Figures

| Figure | File | Content |
|--------|------|---------|
| **A1** | figA1_lspinn_c.png | LS-PINN transmission rate c(t) vs reconstructed reference — rMSE = 1.93e-05 |
| **A2** | figA2_lspinn_SEIR.png | LS-PINN S, E, I, R compartments with ±1σ confidence bands (3 runs) |
| **B** | figB_training_curves.png | Real training loss curves: total loss (Phase 1+2) · loss decomposition (L_data, L_phys, L_IC) · NTK weights (w_S, w_I, w_E) |
| **C1** | figC1_comparison_c.png | LS-PINN vs PINN-std — c(t) (both converge to same curve — shared loss) |
| **C2** | figC2_comparison_SEIR.png | LS-PINN vs PINN-std — S, E, I, R with ±1σ bands (key difference: E and I) |
| **D** | figD_nmpc_comparison.png | NMPC controlled vs uncontrolled (u=0) — all compartments from t=0 with control-start marker at day 30 |

---

## Reproducibility

| Item | Detail |
|------|--------|
| Random seeds | 42, 43, 44 for 3 independent runs |
| Collocation sampling | log-uniform on [0, T_F] — densifies points near t = 0 |
| E(t) strategy | Single signal: same Ẽ(t) reference for both supervision loss and rMSE computation |
| c(t) strategy | Frozen after LS-PINN training — not updated during NMPC |
| Hardware | NVIDIA T4 GPU (Google Colab) |
| Training time | ≈ 190 s/run (Phase 1 + Phase 2) |
| NMPC time | ≈ 300 s (150 IPOPT solves with N = 21 horizon) |

All metrics are reported as **mean ± standard deviation over M = 3 independent runs**.

---

## How to Run

### Google Colab (recommended — NVIDIA T4 GPU)

> ⚠️ Developed and tested on Google Colab with an NVIDIA T4 GPU.
> CPU-only runs are supported but approximately 10–20× slower.

1. Upload `seir_pinn_v8.ipynb` and `maroc_covid_data.xls` to Colab
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run **Cell 1** — configuration, data loading, architecture (~1 min)
4. Run **Cell 2** — training + Figures A, B, C (~20 min for 6 runs)
5. Run **Cell 3** — NMPC + Figure D (~5 min)

Figures are saved to `outputs/figures/`. Download via the Colab file browser.

### Local Installation

```bash
git clone https://github.com/YassineBarakate/seir-pinn-mpc.git
cd seir-pinn-mpc
pip install -r requirements.txt
jupyter notebook seir_pinn_v8.ipynb
```

---

## Requirements

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

## Literature References

| Reference | Role |
|-----------|------|
| Li, Q. et al. (2020). *Early Transmission Dynamics in Wuhan.* NEJM 382, 1199–1207. | Latent period e = 1/5.2 day⁻¹ |
| WHO / CDC COVID-19 clinical guidelines. | Recovery rate γ = 1/14 day⁻¹ |
| World Bank (2020). *World Development Indicators.* | Birth/death rate b = d = 1/(70×365) day⁻¹ |
| Haut-Commissariat au Plan (HCP), Morocco (2020). | Total population N_pop = 37×10⁶ |
| Alimohamadi, Y. et al. (2020). *Estimate of the Basic Reproduction Number for COVID-19.* J. Prev. Med. Public Health 53(3), 151–157. | R₀ range validation |
| Ait Mahiout, L. et al. (2020). *Impact Assessment of Containment Measure Against COVID-19 in Morocco.* Chaos Solitons Fractals 140, 110231. | Morocco-specific epidemiological context |
| Anderson, R. M. & May, R. M. (1991). *Infectious Diseases of Humans.* Oxford University Press. | Vaccination sensitivity modeling |
| Johns Hopkins University CSSE (2023). *COVID-19 Data Repository.* github.com/CSSEGISandData/COVID-19 | Case data anchor |
| Our World in Data (2023). *COVID-19 Vaccination Data.* ourworldindata.org | Vaccination time series U(t) |
| Kaipio, J. & Somersalo, E. (2005). *Statistical and Computational Inverse Problems.* Springer. | Inverse problem methodology |
| Grimm, V. et al. (2022). *Estimating the time-dependent contact rate of SIR and SEIR models using PINNs.* ETNA 56, 1–27. | Precedent for PINN identification |
| Millevoi, C. et al. (2024). *A PINN approach for compartmental epidemiological models.* PLOS Comput. Biol. | Precedent for partial observation in SEIR-PINN |
| Zhong, A., She, B. & Paré, P. E. (2025). *A PINNs-Based MPC Framework for SIR Epidemics.* arXiv:2509.12226. | Precedent for PINN + MPC pipeline |

---

## Citation

If you use this code or the associated methodology, please cite:

```bibtex
@article{khazri2026seir,
  title     = {Parametric Identification of Time-Varying Transmission and Vaccination
               Control in SEIR Epidemics via Physics-Informed Neural Networks},
  author    = {Khazri, Omar and Barakate, Yassine},
  year      = {2026},
  institution = {Faculty of Sciences Ben M'Sik, Hassan II University of Casablanca}
}
```

---

## License

License will be added upon publication of the accompanying paper.
© 2026 Omar Khazri, Yassine Barakate — Hassan II University of Casablanca, Morocco.
Please contact the authors before reusing this code.
