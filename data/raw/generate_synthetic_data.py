"""
Synthetic Data Generator for Emerging-Market Credit Decisioning
==============================================================
Generates a realistic synthetic dataset with:
- Traditional tabular features
- Alternative data (mobile money, utility payments, etc.)
- Sequential transaction tensors suitable for CNN encoding
- Thin-file flag and default labels

This script produces files that can be placed in data/raw/ and later cleaned
into data/processed/.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
N_SAMPLES = 48000
SEQ_LEN = 30          # days
N_SEQ_FEATURES = 12   # transaction channels
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
OUT_DIR = Path(__file__).parent


def generate_data(n_samples: int = N_SAMPLES):
    print(f"Generating {n_samples:,} synthetic applicant records...")

    # --- Identifiers & basic demographics ---
    applicant_id = [f"APP_{i:06d}" for i in range(n_samples)]
    age = np.random.randint(21, 65, n_samples)
    gender = np.random.choice(["M", "F"], n_samples, p=[0.55, 0.45])

    # --- Thin-file status (≈38%) ---
    thin_file_flag = np.random.binomial(1, 0.38, n_samples)

    # --- Traditional features (more missing for thin-file) ---
    bureau_score = np.where(
        thin_file_flag == 1,
        np.nan,
        np.clip(np.random.normal(650, 80, n_samples), 300, 850)
    )
    existing_loans = np.where(
        thin_file_flag == 1,
        np.random.poisson(0.3, n_samples),
        np.random.poisson(1.8, n_samples)
    )

    # --- Alternative data ---
    mobile_money_ratio = np.clip(np.random.beta(2, 5, n_samples), 0, 1)
    util_payment_consistency = np.clip(np.random.beta(5, 2, n_samples), 0, 1)
    # Make utility consistency lower for future defaulters
    income_proxy = np.exp(np.random.normal(8.5, 0.7, n_samples))  # log-normal

    # --- Sequential transaction tensor (N, SEQ_LEN, N_SEQ_FEATURES) ---
    # Simple generative process: base level + trend + noise + default-related spike
    txn_seq = np.random.randn(n_samples, SEQ_LEN, N_SEQ_FEATURES).astype(np.float32) * 0.5

    # Add some structure
    for i in range(n_samples):
        trend = np.linspace(0, np.random.uniform(-0.5, 0.8), SEQ_LEN)
        txn_seq[i, :, 0] += trend                         # inflow channel
        txn_seq[i, :, 1] += np.random.uniform(0.2, 1.0)  # outflow baseline

    # --- Default label (≈12%) with dependence on features ---
    logit = (
        -2.8
        + 0.015 * (700 - np.nan_to_num(bureau_score, nan=550))
        - 1.8 * util_payment_consistency
        - 0.9 * mobile_money_ratio
        + 0.4 * thin_file_flag
        + np.random.normal(0, 0.6, n_samples)
    )
    default_prob = 1 / (1 + np.exp(-logit))
    default_12m = (np.random.rand(n_samples) < default_prob).astype(int)

    # Make recent transaction patterns slightly different for defaulters
    defaulters = default_12m == 1
    txn_seq[defaulters, -7:, 0] -= 0.6   # lower recent inflows
    txn_seq[defaulters, -7:, 1] += 0.4   # higher recent outflows

    # --- Assemble tabular DataFrame ---
    df = pd.DataFrame({
        "applicant_id": applicant_id,
        "age": age,
        "gender": gender,
        "thin_file_flag": thin_file_flag,
        "bureau_score": bureau_score,
        "existing_loans": existing_loans,
        "mobile_money_ratio": mobile_money_ratio,
        "util_payment_consistency": util_payment_consistency,
        "income_proxy": income_proxy,
        "default_12m": default_12m,
    })

    print(f"Default rate: {df['default_12m'].mean():.2%}")
    print(f"Thin-file rate: {df['thin_file_flag'].mean():.2%}")

    return df, txn_seq


if __name__ == "__main__":
    df, txn_seq = generate_data()

    # Save
    tabular_path = OUT_DIR / "synthetic_applicants.csv"
    seq_path = OUT_DIR / "transaction_sequences.npy"

    df.to_csv(tabular_path, index=False)
    np.save(seq_path, txn_seq)

    print(f"\nSaved:")
    print(f"  Tabular  → {tabular_path}")
    print(f"  Sequences → {seq_path}")
    print("Done.")
