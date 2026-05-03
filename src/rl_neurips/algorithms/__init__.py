"""
AITPR Algorithm Implementations

This module contains implementations of the Adaptive Information-Theoretic 
Policy Regularization (AITPR) method integrated with various RL algorithms.
"""

from .aitpr_ppo import AitprPPO, AitprPolicyNetwork, AitprValueNetwork
from .aitpr_sac import AitprSAC, AitprSACPolicyNetwork, AitprSACCriticNetwork, ReplayBuffer
from .baselines import (
    PPOBaseline, SACBaseline, A2CBaseline, TD3Baseline,
    PPOEntropyBaseline, SACKLBaseline, TRPOBaseline,
    create_baseline_agent
)

__all__ = [
    "AitprPPO",
    "AitprPolicyNetwork", 
    "AitprValueNetwork",
    "AitprSAC",
    "AitprSACPolicyNetwork",
    "AitprSACCriticNetwork", 
    "ReplayBuffer",
    "PPOBaseline",
    "SACBaseline", 
    "A2CBaseline",
    "TD3Baseline",
    "PPOEntropyBaseline",
    "SACKLBaseline",
    "TRPOBaseline",
    "create_baseline_agent",
]