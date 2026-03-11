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

<img width="1107" height="759" alt="image" src="https://github.com/user-attachments/assets/2926e0f5-8c9d-4dad-85b0-edb2b145ad01" />



The framework consists of three sequential steps:
1. **Data generation** via MC ABAQUS (Monte Carlo + FE simulation)
2. **Model training** with the Att-MMDL architecture
3. **Probabilistic prediction** on unseen GM and structural parameter combinations

---

## Architecture

<img width="1501" height="573" alt="image" src="https://github.com/user-attachments/assets/22b11258-edd8-44c5-87af-f054ff777653" />



| Component | Description |
|---|---|
| **GM Encoder** | Residual 1D-CNN — extracts temporal and frequency features from seismic signals |
| **Structural Parameter Encoder** | ANN — encodes scalar inputs (ρ, E, ν, ζ) into latent space |
| **Cross-Modal Attention** | GM features as queries; structural parameters as keys & values — adaptive, time-varying interaction |
| **Decoder** | 1D-CNN with upsampling — reconstructs acceleration responses at 60 node locations |

---

## Target Structure

<img width="933" height="773" alt="image" src="https://github.com/user-attachments/assets/ccabd74a-f6d2-4796-bec7-e61e5d625491" />



The model targets a **6-story NPP auxiliary building** (KAERI reference model):
- 17,233 shell elements (S4R + S3R)
- Linear elastic behavior (APR 1400 SSE design criteria, PGA = 0.3 g)
- Horizontal (x–y) seismic excitation
- Dynamic responses via mode-superposition method (100 modes)

---

## Key Results

<img width="1667" height="886" alt="image" src="https://github.com/user-attachments/assets/871ba729-8da9-4ec6-8dba-28e409705ca9" />



| Model | Scenario 1 mMAPE | Scenario 2 mMAPE | R² |
|---|---|---|---|
| Baseline 1 (Res-1D CNN + addition) | 2.24% | 2.02% | 0.96 |
| Baseline 2 (LSTM + addition) | 3.13% | 2.58% | 0.94 |
| **Att-MMDL (Proposed)** | **1.03%** | **0.95%** | **0.98–0.99** |

- Error reduction of **54%** and **67%** relative to baseline models 1 and 2, respectively
- Inference time: **millisecond scale** (suitable for real-time post-earthquake assessment)

---

## Probabilistic Risk Quantification

<img width="1002" height="777" alt="image" src="https://github.com/user-attachments/assets/6bc7dcf7-302e-4035-9054-6933045ea364" />



The framework computes temporal exceedance probability in real time:

$$P_{exc}(t) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(a_i(t) > a_{\text{threshold}})$$

This transforms binary risk classification into **continuous risk quantification**, enabling prioritization of equipment inspections across node locations — even when deterministic predictions assign identical risk.

---

## Repository Structure

```
Att-MMDL/
├── Baselinemodel_LSTM/
│   ├── LSTM_based_model.m        # Model architecture definition
│   └── net.mat                   # Pretrained weights
│
├── Baselinemodel_Res1DCNN/
│   ├── experimental_layer_v3.m   # Model architecture definition (custom layer)
│   └── net.mat                   # Pretrained weights
│
├── Proposedmodel_RESS/
│   ├── createSeismicAttentionModel_vFinal2.m   # Model architecture definition
│   └── net.mat                                 # Pretrained weights
│
├── Model Train and Models/
│   ├── AEQ_DLMODEL_Trainnet.m    # Main training script
│   ├── Baselinemodel_LSTM.m      # Baseline 2 model definition (for training)
│   ├── Baselinemodel_ResCNN.m    # Baseline 1 model definition (for training)
│   └── Proposedmodel_RESS.m      # Proposed model definition (for training)
│
└── README.md
```

> Note: Training data is not included (confidential NPP simulation data).

---

## How to Load Pretrained Models

> ⚠️ **Important**: Each `net.mat` file must be loaded using the `.m` script located in the **same folder**. **Do not rename the `.m` files** — the model weights stored in `net.mat` depend on the custom layer and architecture definitions in the accompanying `.m` file. Renaming or moving the `.m` file will cause MATLAB to fail when reconstructing the network from the `.mat` file.

### Proposed Model (Att-MMDL)
```matlab
% Step 1. Add the folder to MATLAB path (the .m file must be present)
addpath('Proposedmodel_RESS')

% Step 2. Load pretrained weights
load('Proposedmodel_RESS/net.mat')   % loads variable 'net' into workspace
```

### Baseline 1 (Res-1D CNN)
```matlab
addpath('Baselinemodel_Res1DCNN')
load('Baselinemodel_Res1DCNN/net.mat')
```

### Baseline 2 (LSTM)
```matlab
addpath('Baselinemodel_LSTM')
load('Baselinemodel_LSTM/net.mat')
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
  pages   = {112582},
  year    = {2026},
  note    = {In Press, Available online 10 March 2026},
  doi     = {10.1016/j.ress.2026.112582}
}
```

---

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2022-00144434) and by Korea Research Institute of Standards and Science (KRISS-2025-GP2025-0009).
