# Multi-Agent Deep Reinforcement Learning Framework with CNN-Encoded Alternative Data for Real-Time Credit Decisioning in Emerging Markets

## Overview
This project implements a multi-agent deep reinforcement learning (MARL) system combined with a CNN encoder for sequential alternative data to improve credit decisioning, especially for thin-file customers in emerging markets.

## Research Questions
- **RQ1**: How can a multi-agent deep reinforcement learning framework improve real-time credit decisioning compared to single-agent DRL or traditional ML models in emerging market contexts?
- **RQ2**: To what extent does CNN-based encoding of alternative data enhance feature representation and predictive performance over traditional tabular features?
- **RQ3**: How effectively can the proposed framework adapt to dynamic risk environments while maintaining fairness?
- **RQ4**: What is the impact on key business and inclusion metrics (thin-file approval rates, portfolio expected loss, profitability)?

## Project Structure
```
credit-marl/
├── data/
│   ├── raw/
│   │   └── german.data
│   └── processed/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline_Models.ipynb
│   ├── 03_CNN_Encoder.ipynb
│   ├── 04_DRL_Training.ipynb
│   ├── 05_Feature_Fusion.ipynb
│   ├── 06_Single_Agent_PPO.ipynb
│   ├── 07_Multi_Agent_Training.ipynb
│   ├── 08_MARL_vs_Baselines.ipynb
│   ├── 09_Adaptation_Fairness.ipynb
│   └── 10_Business_Metrics.ipynb
├── results/
├── src/
├── requirements.txt
└── README.md
```

## How to Run
1. Place `german.data` in `data/raw/` (already included).
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks **in order** from `01_EDA.ipynb` → `10_Business_Metrics.ipynb`.
4. All models, metrics, and figures are saved in the `results/` folder.

## Key Components
- **01–02**: Data loading from German Credit, thin-file engineering, traditional baselines
- **03**: 1-D CNN encoder for synthetic transaction sequences
- **05**: Feature fusion (tabular + CNN embeddings)
- **06**: Single-agent PPO baseline
- **07**: Multi-agent system (Risk + Fairness + Portfolio agents)
- **08**: Full comparison table (RQ1)
- **09**: Distribution-shift and fairness evaluation (RQ3)
- **10**: Business metrics – thin-file inclusion & expected loss (RQ4)

## Requirements
See `requirements.txt` for the full list (PyTorch, scikit-learn, pandas, etc.).

## Notes
- The synthetic transaction sequences are generated in `01_EDA.ipynb` to simulate mobile-money / digital-footprint alternative data common in emerging markets.
- Reward design follows the classic German Credit cost matrix with additional thin-file inclusion incentives.
```

