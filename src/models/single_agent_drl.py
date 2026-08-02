"""
Single-Agent Deep Reinforcement Learning for Credit Decisioning
----------------------------------------------------------------
Placeholder for DQN / PPO agent that learns an approve/reject (or continuous limit) policy.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CreditPolicyNetwork(nn.Module):
    """Simple MLP policy / Q-network that can take CNN embedding + tabular features."""

    def __init__(self, state_dim: int, action_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# TODO: Integrate with stable-baselines3 or custom DQN/PPO training loop
# Example usage (to be expanded):
#   from stable_baselines3 import PPO
#   model = PPO("MlpPolicy", env, verbose=1)
#   model.learn(total_timesteps=100_000)
