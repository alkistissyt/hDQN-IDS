# hDQN-IDS

# A Hierarchical Reinforcement Learning Framework for Resource-Aware Intrusion Detection
A reinforcement learning-based intrusion detection system using **hierarchical Deep Q-Networks (h-DQN)** with adaptive computational cost management. This repository supports my bachelor thesis on designing and implementing a hierarchical reinforcement learning (HRL) system for intrusion detection using multiple network security datasets. The framework aims to detect network intrusions while balancing detection performance with computational resource usage.

## 🎯 What This Offers

- **Hierarchical Architecture**: Manager agent selects detection mode, Worker agent makes classification decisions
- **Resource Management**: A Manager agent is introduced to handle intelligent resource allocation. Based on real-time system load and threat level assessments, the agent selects between lightweight and intensive analysis modes
- **Multi-Dataset Support**: Works with CICIDS2017, NF-ToN-IoT, Edge-IIoTset, BoT-IoT
- **Adaptive PCA**: Automatically selects optimal feature dimensions based on variance preservation
- **Smart Cost Management**: Dynamic computational costs based on system queue length
- **Enhanced Rewards**: Realistic reward structure for intrusion detection scenarios

## 📂 Project Structure
- `data_preprocessing.py` - Generic preprocessing with adaptive PCA for multiple IDS datasets (CICIDS2017, NF-ToN-IoT, Edge-IIoTset, BoT-IoT)
- `environment.py` - RL environments with hierarchical architecture, smart features, and adaptive computational cost management
- `train.py`  - Complete training pipeline for both worker and manager agents with enhanced metrics tracking
- `Visualization.py` - Publication-ready visualization generation using actual training results and performance data

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries (install via pip):
  
  ```bash
  pip install -r requirements.txt

### Running the Code
Preprocess the data:

 ```bash
python Preprocessing.py
Initiate costum Environment
 ```bash
python Environmpent.py

Train the model:

 ```bash
python Train.py


## 📊 Datasets Supported

| Dataset | Attack Types | Features |
|---------|--------------|----------|
| CICIDS2017 | 14 types | 80+ features |
| NF-ToN-IoT | 9 types | 40+ features |
| Edge-IIoTset | 15 types | 60+ features |
| BoT-IoT | 5 types | 40+ features |

