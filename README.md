🔥# TSPM-Mamba: Traffic Scene Parsing Mamba🔥 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Stars](https://img.shields.io/github/stars/billfjj/TSPM-Mamba)

📋This is the official implementation of "Cross-domain Traffic Scene Parsing via Vision Foundation Models: A Roadside Data Scarcity Solution"
🌟

![GitHub watchers](https://img.shields.io/github/watchers/IronmanVsThanos/ATM-Traffic.svg?style=social)## Overview

Traffic scene parsing from roadside views faces significant challenges due to limited data availability and poor generalization of existing methods. We propose ATM (Adaptive Token Modulator), a novel approach that:

- Efficiently leverages Vision Foundation Models (VFMs) for roadside traffic scene parsing
- Achieves SOTA performance with only 2.5% trainable parameters
- Shows strong generalization capability in zero-shot and few-shot scenarios
- Performs robustly in challenging conditions (night, rain, etc.)

## 📊 Performance
- 🚀**Parameter Efficiency**: Achieves 78.9% mIoU on TSP6K using only 7.7M parameters (2.5% of full model)
- 🚀**Zero-shot Performance**: 
  - Cityscapes: 76.28% mIoU
  - TSP6K: 54.57% mIoU  
  - RS2K: 64.10% mIoU
- 🚀**Few-shot Learning**: With <10% training data achieves:
  - Cityscapes: 78.58% mIoU
  - TSP6K: 62.35% mIoU
  - RS2K: 68.46% mIoU

## 🚀Installation
```bash
# Clone the repository
git clone https://github.com/IronmanVsThanos/ATM-Traffic.git
cd ATM-Traffic
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip in