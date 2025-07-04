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

## 📊 Datasets
This project leverages four publicly available intrusion detection datasets. Each offers different characteristics and attack types suitable for evaluating network-based detection systems. Download them using the links below:

- CIC‑IDS2017 – A
Official download: http://www.unb.ca/cic/datasets/ids-2017.html 
- NF‑ToN‑IoT – NetFlow version of ToN‑IoT, labeled for multiple IoT attack types.
Official download: https://espace.library.uq.edu.au/view/UQ%3A44d7c5e 
- Edge‑IIoTset – Realistic IoT/IIoT dataset with diverse devices and attacks.
Official download (IEEE DataPort): https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications 
unb.ca
- BoT‑IoT – IoT network dataset containing DDoS, DoS, scanning, and more.
Official download (UNSW Canberra/Impact Cyber Trust): https://research.unsw.edu.au/projects/bot-iot-dataset

Datasets Supported

# IoT/IIoT Cybersecurity Datasets

## Available Datasets

# IoT/IIoT Cybersecurity Datasets

| Dataset | Description | Features | Samples | Size | Download |
|---------|-------------|----------|---------|------|----------|
| **CIC-IDS2017** | Network intrusion detection dataset with various attack types (DDoS, DoS, Brute Force, Heartbleed, etc.) | 80 | 2.5M | Large | [UNB](http://www.unb.ca/cic/datasets/ids-2017.html) |
| **NF-ToN-IoT** | NetFlow version of ToN-IoT dataset labeled for IoT attack types | 43 | 12.9M | Medium | [UQ eSpace](https://espace.library.uq.edu.au/view/UQ%3A44d7c5e) |
| **Edge-IIoTset** | Realistic IoT/IIoT dataset with diverse devices and attacks for edge computing | 61 | 2M | Large | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) |
| **BoT-IoT** | Large-scale IoT network dataset with DDoS, DoS, scanning, and botnet attacks | 10 | 600k | Very Large | [UNSW Canberra](https://research.unsw.edu.au/projects/bot-iot-dataset) |



## Usage Notes

- All datasets are publicly available for research purposes
- Check individual dataset licenses and terms of use before downloading
- Consider computational requirements for processing large datasets


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

