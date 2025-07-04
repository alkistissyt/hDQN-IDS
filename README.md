# hDQN-IDS

# A Hierarchical Reinforcement Learning Framework for Resource-Aware Intrusion Detection

This repository supports my bachelor thesis on designing and implementing a **hierarchical reinforcement learning (HRL)** system for intrusion detection using the **CIC-IDS2017** dataset. The framework aims to detect network intrusions while balancing detection performance with computational resource usage.

## 📂 Project Structure

The core of the implementation is structured across four main Python scripts:

- `data_preprocessing.py`: Loads, cleans, and prepares the CIC-IDS2017 dataset.
- `environment.py`: Defines a custom OpenAI Gym environment, including reward logic and interaction with the HRL agent.
- `train.py`: Training pipeline for both worker and manager agents
- `Visualisation.py`: Evaluates the trained model using classification metrics (accuracy, precision, recall, F1-score).

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries (install via pip):
  
  ```bash
  pip install -r requirements.txt

## 🎯 What This Offers

- **Hierarchical Architecture**: Manager agent selects detection mode, Worker agent makes classification decisions
- **Multi-Dataset Support**: Works with CICIDS2017, NF-ToN-IoT, Edge-IIoTset, BoT-IoT
- **Adaptive PCA**: Automatically selects optimal feature dimensions based on variance preservation
- **Smart Cost Management**: Dynamic computational costs based on system queue length
- **Enhanced Rewards**: Realistic reward structure for intrusion detection scenarios

## 🚀 Quick Start



## 📁 Files

- `data_preprocessing.py` - Generic preprocessing with adaptive PCA for multiple IDS datasets
- `environment.py` - RL environments with hierarchical architecture and smart features  
- `train.py` - Training pipeline for both worker and manager agents

## ⚙️ Key Features

### Reward Structure
- Block Malicious: +5.0 | Allow Benign: +4.0
- Allow Malicious: -10.0 | Block Benign: -2.0

### Adaptive Costs
- Lightweight mode: 0.1 + 0.01 × queue_length
- Full mode: 1.0 + 0.02 × queue_length

### Smart Features
- Anomaly score tracking
- Traffic complexity analysis
- Sequential data processing (no shuffling)

## 📊 Datasets Supported

| Dataset | Attack Types | Features |
|---------|--------------|----------|
| CICIDS2017 | 14 types | 80+ features |
| NF-ToN-IoT | 9 types | 40+ features |
| Edge-IIoTset | 15 types | 60+ features |
| BoT-IoT | 5 types | 40+ features |

