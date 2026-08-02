"""
Evaluation metrics for credit decisioning
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def thin_file_approval_rate(y_pred, thin_file_flag):
    """Approval rate among thin-file applicants (inclusion metric)."""
    mask = thin_file_flag == 1
    if mask.sum() == 0:
        return np.nan
    return (y_pred[mask] == 1).mean()


def portfolio_expected_loss(pd, lgd=0.45, ead=1.0):
    """Simple expected loss calculation (can be extended)."""
    return np.sum(pd * lgd * ead)
