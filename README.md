# Att-MMDL

**Predicting Seismic Floor Response for Nuclear Power Plant Structures with Time-series Uncertainty Propagation using Attention-enhanced Multimodal Deep Learning**

> Jingoo Lee¹, Seungjun Lee², Young-Joo Lee¹, Jaebeom Lee²³  
> ¹ UNIST &nbsp;|&nbsp; ² KRISS &nbsp;|&nbsp; ³ University of Science and Technology  
> *Reliability Engineering & System Safety*, In Press (Available online 10 March 2026)  
> https://doi.org/10.1016/j.ress.2026.112582

---

## Overview

This repository provides the model implementation and pretrained weights for the **Att-MMDL** framework. The training data (FE simulation results from a nuclear power plant auxiliary building) is **not publicly available** due to confidentiality.

The Att-MMDL framework is a surrogate model that replaces expensive Monte Carlo Finite Element Analysis (MC-FEA) for **real-time probabilistic seismic response prediction** of NPP structures. It takes two heterogeneous inputs — ground motion (GM) time series and structural parameters — and predicts full acceleration time-histories at 60 structural locations simultaneously.

---

## Framework

<!-- Insert Fig. 1 (Overview of the proposed framework) here -->

The framework consists of three sequential steps:
1. **Data generation** via MC ABAQUS (Monte Carlo + FE simulation)
2. **Model training** with the Att-MMDL architecture
3. **Probabilistic prediction** on unseen GM and structural parameter combinations

---

## Architecture

<!-- Insert Fig. 6 (Proposed network architecture) here -->

| Component | Description |
|---|---|
| **GM Encoder** | Residual 1D-CNN — extracts temporal and frequency features from seismic signals |
| **Structural Parameter Encoder** | ANN — encodes scalar inputs (ρ, E, ν, ζ) into latent space |
| **Cross-Modal Attention** | GM features as queries; structural parameters as keys & values — adaptive, time-varying interaction |
| **Decoder** | 1D-CNN with upsampling — reconstructs acceleration responses at 60 node locations |

---

## Target Structure

<!-- Insert Fig. 3 or Fig. 4 (FEM of the auxiliary building) here -->

The model targets a **6-story NPP auxiliary building** (KAERI reference model):
- 17,233 shell elements (S4R + S3R)
- Linear elastic behavior (APR 1400 SSE design criteria, PGA = 0.3 g)
- Horizontal (x–y) seismic excitation
- Dynamic responses via mode-superposition method (100 modes)

---

## Key Results

<!-- Insert Fig. 9–10 (Scenario 1) and Fig. 13–14 (Scenario 2) here -->

| Model | Scenario 1 mMAPE | Scenario 2 mMAPE | R² |
|---|---|---|---|
| Baseline 1 (Res-1D CNN + addition) | 2.24% | 2.02% | 0.96 |
| Baseline 2 (LSTM + addition) | 3.13% | 2.58% | 0.94 |
| **Att-MMDL (Proposed)** | **1.03%** | **0.95%** | **0.98–0.99** |

- Error reduction of **54%** and **67%** relative to baseline models 1 and 2, respectively
- Inference time: **millisecond scale** (suitable for real-time post-earthquake assessment)

---

## Probabilistic Risk Quantification

<!-- Insert Fig. 17 or Fig. 18 (Deterministic vs. probabilistic prediction) here -->

The framework computes temporal exceedance probability in real time:

$$P_{exc}(t) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(a_i(t) > a_{\text{threshold}})$$

This transforms binary risk classification into **continuous risk quantification**, enabling prioritization of equipment inspections across node locations — even when deterministic predictions assign identical risk.

---

## Repository Structure

```
Att-MMDL/
├── Model Train and Models/
│   ├── train_proposed.m          # Training script — proposed model
│   ├── train_baseline1.m         # Training script — baseline 1
│   ├── train_baseline2.m         # Training script — baseline 2
│   ├── proposed_model.mat        # Pretrained weights — proposed
│   ├── baseline1_model.mat       # Pretrained weights — baseline 1
│   └── baseline2_model.mat       # Pretrained weights — baseline 2
└── README.md
```

> Note: Training data is not included (confidential NPP simulation data).

---

## How to Load Pretrained Models

> ⚠️ **Important**: Each `.mat` file must be opened using the `.m` script located in the **same directory**. **Do not rename the `.m` files** — the scripts rely on exact filenames to locate and load the model weights correctly.

```matlab
% Step 1. Navigate to the directory containing both .mat and .m files
cd('Model Train and Models')

% Step 2. Run the loading script (do NOT rename this file)
run('load_proposed.m')   % for proposed model
run('load_baseline1.m')  % for baseline 1
run('load_baseline2.m')  % for baseline 2
```

---

## Inputs / Outputs

**Inputs:**
- Ground motion sequence: `[3000 × 1]` — 30 s duration at 100 Hz sampling rate
- Structural parameters: `[4 × 1]` — density (ρ), Young's modulus (E), Poisson's ratio (ν), modal damping ratio (ζ) of concrete

**Output:**
- Acceleration time-history at 60 structural nodes: `[3000 × 60]`

---

## Citation

```bibtex
@article{lee2026attmmdl,
  title   = {Predicting Seismic Floor Response for Nuclear Power Plant Structures 
             with Time-series Uncertainty Propagation Using Attention-enhanced 
             Multimodal Deep Learning},
  author  = {Lee, Jingoo and Lee, Seungjun and Lee, Young-Joo and Lee, Jaebeom},
  journal = {Reliability Engineering \& System Safety},
  volume  = {},
  pages   = {112582},
  year    = {2026},
  note    = {In Press},
  doi     = {10.1016/j.ress.2026.112582}
}
```

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2022-00144434) and by Korea Research Institute of Standards and Science (KRISS-2025-GP2025-0009).
