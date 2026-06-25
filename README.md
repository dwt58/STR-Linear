# STR-Linear

> A linear-based high-performance lightweight model for multivariate time series forecasting


This repository is the official PyTorch implementation of the paper **"STR-Linear: A linear-based high-performance lightweight model for multivariate time series forecasting"** (Knowledge-Based Systems, 2026).

---

## Overview

Multivariate Time Series Forecasting (MTSF) is challenging due to the intertwining of temporal dependencies within each channel and spatial correlations across multiple channels. Existing Linear-based models employing Channel Independent Modeling (CIM) fail to capture multivariate correlations, while Transformer-based models suffer from huge parameter overheads.

**STR-Linear** proposes a **"spatial-temporal divide-and-conquer"** philosophy to address this trade-off:

- It uses **CycleNet** as the temporal backbone to explicitly capture long-term trends and periodic patterns within each channel.
- It introduces a novel **Spatio-Temporal Refiner (STR)** as a lightweight post-processing module. STR leverages cross-channel multi-scale dilated convolutions and attention mechanisms to refine the initial predictions by capturing complex multivariate correlations.

**Key achievement**: STR-Linear reduces parameters by **93.8%** compared to iTransformer, while maintaining competitive or superior prediction accuracy across multiple real-world datasets.

![](https://github.com/dwt58/STR-Linear/blob/f8feed490b4f296b41f00ee0a9937d7c009385c9/Figures/Figure%201.png)
---

## Core Contributions

1.  **High-performance lightweight MTSF model**  
    We propose STR-Linear, which uses a Linear layer as the backbone prediction network. It overcomes the inherent performance bottleneck of Linear-based models in MTSF while maintaining an ultra-low parameter count.

2.  **Spatio-Temporal Refiner (STR) as a post-processing strategy**  
    Unlike conventional models that intertwine spatial and temporal modeling, we decouple them. STR serves as a plug-and-play post-hoc refinement module, compensating for the missing inter-variable correlations without interfering with the temporal backbone.

3.  **Excellent performance-efficiency balance**  
    STR-Linear achieves state-of-the-art competitive results on traffic and energy datasets. It significantly outperforms existing linear models and matches or exceeds complex Transformer/GNN models with drastically lower computational cost.

---

## Key Components

### 1. Temporal Backbone (CycleNet)
- Utilizes the **Residual Cycle Forecasting (RCF)** module to explicitly model periodic patterns.
- Decomposes the raw series into **periodic** and **trend** components, which are predicted separately and then summed to produce the initial forecast \( y_{raw} \).

### 2. Spatio-Temporal Refiner (STR)
The STR module captures spatial dependencies missed by the backbone through three steps:

- **Feature Encoding & Channel Projection**: Maps the initial predictions into a high-dimensional latent space.
- **Multi-Scale Dilated Convolutions**: Uses cascaded convolution branches with dilation rates **[1, 2, 4, 8]** to capture hierarchical spatial receptive fields (from local neighbors to global channels).
- **Grouped Convolution & Feature Recalibration**: Employs grouped convolutions for efficiency and an SE-style attention mechanism to enhance important feature channels.

### 3. Residual Refinement
Instead of reconstructing the entire sequence, STR learns **residual signals** \( y_{refined} \). The final prediction is \( y = y_{raw} + y_{refined} \), preserving the robust temporal patterns from the backbone while adding precise spatial calibrations.

![](https://github.com/dwt58/STR-Linear/blob/58249144c8cec33c0c37bed919ce3fffc0049899/Figures/Figure%203.png)
---


## Experimental Results

### Main Results
We compare STR-Linear against recent state‑of‑the‑art models (CDM, CIM, and HCM) on eight datasets. The table below shows average MSE/MAE over all prediction horizons for representative datasets.

![](https://github.com/dwt58/STR-Linear/blob/main/Figures/Table%202.png)

### Ablation Studies

### Component Ablation (RCF, Temporal Backbone, STR)
We evaluate the contribution of each core component. The table reports averaged results on Traffic, PEMS03, and PEMS04.

![](https://github.com/dwt58/STR-Linear/blob/e0fbd20c862283dde3855f7c0742bcbda7cbe18a/Figures/Table%205.png)

Complete model (V) achieves the best overall performance, confirming that all three components cooperate effectively.


### STR Placement Analysis
Where should the STR module be inserted? We test five positions:

A: Pre‑processing (before backbone)
B: Periodic component refinement
C: Trend component refinement
D: Post‑processing (after backbone) – our default STR‑Linear
w/o: Original CycleNet (no STR)

![](https://github.com/dwt58/STR-Linear/blob/e0fbd20c862283dde3855f7c0742bcbda7cbe18a/Figures/Table%206.png)

## Compatibility Analysis of the STR Module with Other Models
To verify its general applicability, we integrated STR into the classic linear model DLinear and the Transformer-based model iTransformer.

![](https://github.com/dwt58/STR-Linear/blob/101ebe87e32ef9c87a1c1d32c1809466feab7c2c/Figures/Table%2011.png)

## Citation
If you find this work useful, please cite:
```bibtex
@article{deng2026strlinear,
  title={STR-Linear: A linear-based high-performance lightweight model for multivariate time series forecasting},
  author={Deng, Weitao and Tang, Shaomin},
  journal={Knowledge-Based Systems},
  volume={349},
  pages={116442},
  year={2026},
  publisher={Elsevier}
}
