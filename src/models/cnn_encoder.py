"""
CNN Encoder for Sequential Alternative Data
-------------------------------------------
Takes a transaction sequence tensor of shape (batch, seq_len, n_features)
and produces a fixed-size embedding suitable for concatenation with tabular features
or for use as state representation in DRL agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransactionCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        seq_len: int = 30,
        embedding_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, in_channels)  →  we transpose to (batch, channels, seq_len)
        returns: (batch, embedding_dim)
        """
        x = x.transpose(1, 2)               # (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)        # (B, 64)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Quick sanity check
    model = TransactionCNNEncoder()
    dummy = torch.randn(8, 30, 12)
    out = model(dummy)
    print("Output shape:", out.shape)  # expected (8, 64)
