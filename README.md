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

---


## Experimental Results
![](https://github.com/dwt58/STR-Linear/blob/main/Figures/Table%202.png)


## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/dwt58/STR-Linear.git
cd STR-Linear
pip install -r requirements.txt
