"""
Adaptive Information-Theoretic Policy Regularization (AITPR)

AITPR is a novel reinforcement learning method that regularizes policy optimization
using mutual information between policy and value function representations,
with adaptive weighting based on environment reward structure.

Key components:
- AITPR-PPO: PPO with information-theoretic regularization
- AITPR-SAC: SAC with information-theoretic regularization  
- Theoretical bounds and convergence guarantees
- Mutual information estimation methods
- Comprehensive experimental evaluation

Usage:
    from rl_neurips import AitprPPO
    
    agent = AitprPPO(env=env, lambda_base=1.0, alpha=0.1, mi_method="mine")
    episode_info = agent.train_episode()
"""

__version__ = "0.1.0"
__author__ = "Salim Khazem"
__email__ = "salimkhazem97@gmail.com"

from .algorithms import AitprPPO, AitprPolicyNetwork, AitprValueNetwork
from .theory import AitprTheory, AitprMutualInformation, compute_reward_entropy

__all__ = [
    "AitprPPO",
    "AitprPolicyNetwork",
    "AitprValueNetwork", 
    "AitprTheory",
    "AitprMutualInformation",
    "compute_reward_entropy",
]


def main() -> None:
    """Main entry point for the rl-neurips package."""
    print(f"AITPR v{__version__} - Adaptive Information-Theoretic Policy Regularization")
    print(f"Author: {__author__} <{__email__}>")
    print("\nAvailable components:")
    print("- AitprPPO: PPO with information-theoretic regularization")
    print("- AitprSAC: SAC with information-theoretic regularization") 
    print("- AitprTheory: Theoretical bounds and analysis")
    print("- AitprMutualInformation: MI estimation methods")
    print("\nFor usage examples, see the scripts/ directory.")
