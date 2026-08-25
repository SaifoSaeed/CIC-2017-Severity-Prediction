# CIC-2017-Severity-Prediction

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

## Overview
This repository contains the implementation for predicting the severity of network intrusions using the **CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection System) dataset. The project leverages machine learning and deep learning methodologies to classify network traffic flows and assess the severity of potential cyber-attacks, moving beyond standard binary or multi-class detection into risk-based severity scoring.

## Dataset
The [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) includes benign traffic and the most common up-to-date attacks. 
Network traffic was captured via PCAP and processed using **CICFlowMeter** to extract 80+ time-based statistical features (e.g., flow duration, packet length variance, inter-arrival times).

### Data Preprocessing Pipeline
- **Cleaning**: Removal of `NaN` and `Infinity` values standard in raw CICFlowMeter outputs.
- **Normalization**: Application of `StandardScaler` / `MinMaxScaler` for gradient stability during deep learning optimization.
- **Feature Selection**: Dimensionality reduction to eliminate zero-variance and highly correlated features, optimizing computational overhead.

## Repository Structure
```text
├── data/                  # Directory for CIC-2017 CSV files
├── notebooks/             # Exploratory Data Analysis (EDA) and prototyping
├── src/
│   ├── data_loader.py     # Data parsing and preprocessing scripts
│   ├── model.py           # Neural network architecture / ML models
│   ├── train.py           # Training loop and gradient optimization logic
│   └── evaluate.py        # Inference and metrics calculation
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

## Setup & Installation

It is recommended to run this project in an isolated virtual environment (e.g., using `venv` or `conda` on Linux).

```bash
# Clone the repository
git clone https://github.com/SaifoSaeed/CIC-2017-Severity-Prediction.git
cd CIC-2017-Severity-Prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Key packages required (see `requirements.txt` for exact versions):
- `numpy`, `pandas` for low-level matrix manipulation and data processing
- `scikit-learn` for preprocessing and baseline models
- `torch` (PyTorch) for deep neural network execution
- `matplotlib`, `seaborn` for confusion matrix and metric visualization

## Usage

**1. Data Preparation:**
Place the processed CIC-2017 CSV files into the `data/` directory. Run the preprocessing script to clean the data and generate feature tensors:
```bash
python src/data_loader.py --input data/raw --output data/processed
```

**2. Model Training:**
Execute the training script. You can specify hyperparameters such as batch size, learning rate, and epochs.
```bash
python src/train.py --epochs 50 --batch_size 256 --lr 0.001
```

**3. Evaluation:**
Evaluate the trained weights against the test split to compute Precision, Recall, F1-Score, and Severity mapping:
```bash
python src/evaluate.py --model_path weights/best_model.pth
```

## Methodology & Architecture
The model relies on mapping the multi-class attack labels (e.g., DDoS, PortScan, Bot, Web Attack) into distinct **severity tiers** (Low, Medium, High, Critical) based on network impact. The architecture leverages deep learning (via PyTorch) optimized for structured tabular data, implementing robust regularization to prevent overfitting on the majority classes (Benign traffic).

## Contributing
Contributions are welcome. Please submit a pull request detailing the bug fix, feature addition, or performance optimization.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
