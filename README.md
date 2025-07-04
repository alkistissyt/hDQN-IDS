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
This project leverages four publicly available intrusion detection datasets. Download them using the links below:

-CIC‑IDS2017 – Benign and modern attack traffic, in CSV/PCAP formats.
Official download: http://www.unb.ca/cic/datasets/ids-2017.html 

-NF‑ToN‑IoT – NetFlow version of ToN‑IoT, labeled for multiple IoT attack types.
Official download: https://espace.library.uq.edu.au/view/UQ%3A44d7c5e 

-Edge‑IIoTset – Realistic IoT/IIoT dataset with diverse devices and attacks.
Official download (IEEE DataPort): https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications 
unb.ca

-BoT‑IoT – IoT network dataset containing DDoS, DoS, scanning, and more.
Official download (UNSW Canberra/Impact Cyber Trust): https://research.unsw.edu.au/projects/bot-iot-dataset 

Datasets
This project uses four publicly available intrusion detection datasets. Each offers different characteristics and attack types suitable for evaluating network-based detection systems.

🔹 CIC‑IDS2017
A modern dataset containing labeled benign and attack traffic captured in realistic network conditions, including DoS, DDoS, brute force, infiltration, and web-based attacks.
📁 Formats: CSV, PCAP
📍 Source: Canadian Institute for Cybersecurity (UNB)

🔹 NF‑ToN‑IoT
A NetFlow-based version of the ToN‑IoT dataset tailored for IoT networks. Includes multi-class attack labels (e.g., DDoS, injection, ransomware, backdoor) and represents industrial and consumer IoT environments.
📁 Formats: NetFlow CSV
📍 Source: University of Queensland

🔹 Edge‑IIoTset
A comprehensive and realistic cybersecurity dataset for edge and industrial IoT (IIoT) applications. Covers a wide range of attacks across multiple edge devices with time-sequenced behavior.
📁 Formats: CSV
📍 Source: IEEE DataPort

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

