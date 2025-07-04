 # hDQN-IDS

# A Hierarchical Reinforcement Learning Framework for Resource-Aware Intrusion Detection
A reinforcement learning-based intrusion detection system using **hierarchical Deep Q-Networks (h-DQN)** with adaptive computational cost management. This repository supports my bachelor thesis on designing and implementing a hierarchical reinforcement learning (HRL) system for intrusion detection using multiple network security datasets. The framework aims to detect network intrusions while balancing detection performance with computational resource usage.
 
## Project Structure
- `Preprocessing.py` - Generic preprocessing with adaptive PCA for multiple IDS datasets 
- `Environment.py` - RL environments with hierarchical architecture, smart features, and adaptive computational cost management
- `train.py`  - Complete training pipeline for both worker and manager agents with enhanced metrics tracking
- `Visualization.py` - Publication-ready visualization generation using actual training results and performance data

## Datasets
This project leverages four publicly available intrusion detection datasets. Each offers different characteristics and attack types suitable for evaluating network-based detection systems. Download them using the links below:

| Dataset | Description | Features | Samples | Size | Download |
|---------|-------------|----------|---------|------|----------|
| **CIC-IDS2017** | Network intrusion detection dataset with various attack types (DDoS, DoS, Brute Force, Heartbleed, etc.) | 80 | 2.5M | Large | [UNB](http://www.unb.ca/cic/datasets/ids-2017.html) |
| **NF-ToN-IoT** | NetFlow version of ToN-IoT dataset labeled for IoT attack types | 43 | 12.9M | Very Large | [UQ eSpace](https://espace.library.uq.edu.au/view/UQ%3A44d7c5e) |
| **Edge-IIoTset** | Realistic IoT/IIoT dataset with diverse devices and attacks for edge computing | 61 | 2M | Large | [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications) |
| **BoT-IoT** | Large-scale IoT network dataset with DDoS, DoS, scanning, and botnet attacks | 10 | 600k | Small | [UNSW Canberra](https://research.unsw.edu.au/projects/bot-iot-dataset) |

##  Usage

Follow the steps below to run the full pipeline.

### 1. Clone the Repository 
```
bash git clone https://github.com/alkistissyt/hDQN-IDS.git
cd alkistissyt/hDQN-IDS
```
### 2. Install Dependencies
Make sure Python 3.8+ is installed, then run: ``` pip install -r requirements.txt ```

### 3. Prepare the Datasets
Download the datasets listed in the Datasets section and place the raw files (CSV) into the `data/` directory.

Then preprocess the data: ``` python data_preprocessing.py ```

### 4. Initialize the Custom Environment
``` python Environment.py ```

### 5. Train the Model
Run the training script for the Hierarchical Deep Q-Network (h-DQN): ``` python train.py ```

You'll receive evaluation metrics like accuracy, precision, recall, and F1-score.

### 5. Visualize Results (Optional)
To generate reward curves, confusion matrices, or other performance plots: ``` python Visualization.py ```

## Usage Notes

- All datasets are publicly available for research purposes
- Check individual dataset licenses and terms of use before downloading
- Consider computational requirements for processing large datasets
- All trained models will be saved in the `models/` folder.

