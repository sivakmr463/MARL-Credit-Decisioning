# MARL-Credit-Decisioning

**A Multi-Agent Deep Reinforcement Learning Framework with CNN-Encoded Alternative Data for Real-Time Credit Decisioning in Emerging Markets**

This repository contains the code, data, and notebooks for the QM640 Data Analytics Capstone project.

---

## Project Overview

This research designs and evaluates a **multi-agent deep reinforcement learning (MARL)** framework that uses **CNN-encoded alternative data** (transaction sequences, behavioral patterns, digital footprints) for real-time credit decisioning in emerging markets.

### Research Questions

1. **RQ1**: How can a multi-agent DRL framework improve real-time credit decisioning compared to single-agent DRL or traditional ML models in emerging market contexts?
2. **RQ2**: To what extent does CNN-based encoding of alternative data enhance feature representation and predictive performance over traditional tabular features?
3. **RQ3**: How effectively can the framework adapt to dynamic risk environments while maintaining regulatory compliance and fairness?
4. **RQ4**: What is the impact on key business and inclusion metrics (default accuracy, thin-file approval rates, portfolio profitability)?

---

## Repository Structure

```
MARL-Credit-Decisioning/
├── data/
│   ├── raw/                  # Original / synthetic generation scripts & seed data
│   └── processed/            # Cleaned tabular features + sequence tensors
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline_Models.ipynb
│   ├── 03_CNN_Encoder.ipynb
│   └── 04_DRL_Training.ipynb
├── src/
│   ├── models/
│   │   ├── cnn_encoder.py
│   │   ├── single_agent_drl.py
│   │   └── multi_agent_drl.py
│   └── utils/
│       ├── data_loader.py
│       └── metrics.py
├── results/                  # Metrics, figures, model checkpoints
├── docs/                     # Additional documentation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sivakmr463/MARL-Credit-Decisioning.git
cd MARL-Credit-Decisioning
```

### 2. Create environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Generate synthetic data (if not already present)
```bash
python data/raw/generate_synthetic_data.py
```

### 4. Run notebooks
Open the notebooks in order using Jupyter Lab or VS Code.

---

## Data Note

All applicant records are **synthetically generated** to reflect realistic marginal distributions, sequential correlation structures, and missingness patterns observed in published emerging-market alternative-data studies.  
**No real customer data** are stored or processed.

---

## Current Status (Interim Report)

| Stage                    | Status      |
|--------------------------|-------------|
| Data generation & cleaning | Completed |
| Exploratory Data Analysis | Completed |
| Traditional ML baselines  | Completed |
| CNN Encoder               | In Progress |
| Single-agent DRL          | In Progress |
| Multi-agent DRL           | Pending     |
| Fairness & shift experiments | Pending  |
| Final evaluation & report | Pending     |

---

## Citation

If you use this work, please cite:

> [Student Name]. (2026). *A Multi-Agent Deep Reinforcement Learning Framework with CNN-Encoded Alternative Data for Real-Time Credit Decisioning in Emerging Markets*. QM640 Data Analytics Capstone, Walsh College.

---

## License

This project is for academic purposes (QM640 Capstone).  
Please contact the author before using the code or data for any other purpose.
