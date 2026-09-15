# Multi-Agent Deep Reinforcement Learning Framework with CNN-Encoded Alternative Data for Real-Time Credit Decisioning in Emerging Markets

## Overview
This project implements a multi-agent deep reinforcement learning (MARL) system with specialized agents, attention-based coordination, multi-level credit assignment (MACA), and CNN-encoded alternative data for improved credit decisioning — especially for thin-file customers.

## Research Questions
- **RQ1**: Does multi-agent RL with specialized agents outperform single-agent DRL and traditional ML?
- **RQ2**: How much does CNN encoding of sequential alternative data improve performance?
- **RQ3**: Can the framework adapt under distribution shift while preserving fairness?
- **RQ4**: What is the impact on thin-file inclusion, expected loss, and profitability?

## Architecture Highlights
- 4 Specialized Agents: Income, Behavior, Macro, Compliance
- Attention-based Coordinator
- Multi-level Credit Assignment (MACA)
- 1D CNN Encoder + DeepSVDD anomaly filter
- Full explainability via attention weights



## Key Notebooks
- `03_CNN_Encoder.ipynb` → CNN + DeepSVDD
- `07_Multi_Agent_Training.ipynb` → Specialized Agents + MACA training
- `08_MARL_vs_Baselines.ipynb` → Full comparison & ablation
- `09_Adaptation_Fairness.ipynb` → Shift + fairness
- `10_Business_Metrics.ipynb` → Inclusion & loss metrics