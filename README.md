# Deep Neural Networks for Brain Age Prediction from Low-density EEG

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Under_Review-orange)

## Overview

This repository contains the official implementation of the paper: **"Deep Neural Networks for Brain Age Prediction from Low density EEG"**.

We present a computational pipeline that estimates Brain Age using accessible, 8-channel resting-state EEG (rsEEG). The workflow transforms preprocessed EEG signals into Time-Frequency scalograms (using Continuous Wavelet Transform) and processes them through a custom Deep Convolutional Neural Network (CNN) designed for volumetric data.

### Key Features
- **Sparse Montage:** Optimization for 8-channel setups (FP1, FP2, C3, C4, P7, P8, O1, O2).
- **Hybrid Processing:** Scalogram tensor generation and Custom Residual CNN architecture.
- **Clinical Validation:** Tested on Healthy Controls, MCI, AD, and FTD cohorts.

---

## Repository Structure

- `external/APPLEE`: References the automated preprocessing pipeline used for artifact removal (Zapata et al., 2024).
- `processing/`: Code to convert cleaned EEG (`.fif`) into 3D scalogram tensors (`.h5`).
- `model/`: The custom CNN architecture (Residual blocks + SpatialDropout2D) and training scripts.

---

## Installation

We recommend using Anaconda to manage the dual-framework requirements (PyTorch for data processing, TensorFlow for modeling).

```bash
# Clone the repository
git clone [https://github.com/AntoniaYQ/BrainAge-LowDensityEEG.git](https://github.com/AntoniaYQ/BrainAge-LowDensityEEG.git)
cd BrainAge-LowDensityEEG

# Create environment
conda create -n brainage python=3.9
conda activate brainage

# Install dependencies
pip install -r requirements.txt