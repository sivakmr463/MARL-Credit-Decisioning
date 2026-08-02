"""
Multi-Agent Deep Reinforcement Learning Framework
-------------------------------------------------
Core contribution of the project.

Agents (planned):
1. Risk Assessment Agent   – focuses on default probability / expected loss
2. Fairness / Compliance Agent – monitors disparate impact and regulatory constraints
3. Portfolio Optimization Agent – balances long-term profitability and concentration risk

Coordination can be achieved via:
- Shared critic / value function
- Communication channels
- Hierarchical controller
"""

import torch
import torch.nn as nn
from typing import Dict


class RiskAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class FairnessAgent(nn.Module):
    """Placeholder – will output adjustment signals or constraint violations."""
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # fairness score / penalty
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PortfolioAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class MultiAgentCoordinator(nn.Module):
    """
    Simple coordinator that combines outputs from specialized agents.
    This is a starting point – more sophisticated communication protocols
    will be implemented later.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.risk_agent = RiskAgent(state_dim)
        self.fairness_agent = FairnessAgent(state_dim)
        self.portfolio_agent = PortfolioAgent(state_dim)
        self.combine = nn.Linear(2 + 1 + 2, 2)  # risk(2) + fair(1) + port(2)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        risk_out = self.risk_agent(state)
        fair_out = self.fairness_agent(state)
        port_out = self.portfolio_agent(state)
        combined = torch.cat([risk_out, fair_out, port_out], dim=-1)
        final_action_logits = self.combine(combined)
        return {
            "action_logits": final_action_logits,
            "risk": risk_out,
            "fairness": fair_out,
            "portfolio": port_out,
        }
