# Multi-Agent Deep Reinforcement Learning Framework with CNN-Encoded Alternative Data for Real-Time Credit Decisioning in Emerging Markets

## Overview
This project implements a multi-agent deep reinforcement learning (MARL) system with specialized agents, attention-based coordination, multi-level credit assignment (MACA), and CNN-encoded alternative data for improved credit decisioning — especially for thin-file customers.

## Research Questions
- **RQ1**: Does multi-agent RL with specialized agents outperform single-agent DRL and traditional ML?
- **RQ2**: How much does CNN encoding of sequential alternative data improve performance?
- **RQ3**: Can the framework adapt under distribution shift while preserving fairness?
- **RQ4**: What is the impact on thin-file inclusion, expected loss, and profitability?


## Data Sources: The foundation is laid by two distinct data streams:

Tabular Features (German.data): Traditional, static customer information. This includes details like demographics, credit history, purpose of loan, and financial status.

Synthetic 8-Channel Sequences: Dynamically generated time-series data, simulating behavioral patterns across 8 different categories (e.g., mobile money, utility payments) for each customer.

Feature Engineering & Encoding: These raw data sources undergo specialized processing:

Processed Tabular Features: Tabular data is cleaned, preprocessed (e.g., scaling, one-hot encoding), and made ready for modeling.
CNN Encoder: The 8-channel sequences are fed into a Convolutional Neural Network (CNN) specifically designed to extract meaningful temporal patterns and cross-channel interactions.
CNN Embeddings: The output of the CNN Encoder, representing a dense, low-dimensional vector that encapsulates the complex dynamic behaviors from the sequences.
Feature Fusion: This critical step combines the different data representations:

## Concatenate: The Processed Tabular Features and the CNN Embeddings are concatenated (joined side-by-side) to create a single, comprehensive feature vector.
Fused Features: The resulting rich feature set that combines both static, traditional information and dynamic, alternative data insights.
Core Model (MARL): The Fused Features serve as the state input to the Hierarchical Multi-Agent Deep Reinforcement Learning (MARL) Framework, which is the core innovative component of this project.

## Project Output & Impact: The ultimate outcomes of the MARL framework:

Real-Time Credit Decisions (Approve/Reject/Counter): The primary output, representing the intelligent decisions made by the MARL agents based on the comprehensive fused features.
Optimized Business Metrics & Portfolio Impact: The aggregate effects of these decisions, leading to improved profitability, reduced expected losses, and enhanced financial inclusion, which are the overarching goals of the project.