# hDQN-IDS

# A Hierarchical Reinforcement Learning Framework for Resource-Aware Intrusion Detection

This repository supports my bachelor thesis on designing and implementing a **hierarchical reinforcement learning (HRL)** system for intrusion detection using the **CIC-IDS2017** dataset. The framework aims to detect network intrusions while balancing detection performance with computational resource usage.

## 📂 Project Structure

The core of the implementation is structured across four main Python scripts:

- `data_preprocessing.py`: Loads, cleans, and prepares the CIC-IDS2017 dataset.
- `environment.py`: Defines a custom OpenAI Gym environment, including reward logic and interaction with the HRL agent.
- `train.py`: Trains a Hierarchical Deep Q-Network (h-DQN) model using the custom environment.
- `evaluate.py`: Evaluates the trained model using classification metrics (accuracy, precision, recall, F1-score).

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required libraries (install via pip):
  
  ```bash
  pip install -r requirements.txt
