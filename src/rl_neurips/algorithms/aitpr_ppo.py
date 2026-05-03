"""
AITPR-PPO: Adaptive Information-Theoretic Policy Regularization with PPO

This module implements the AITPR method integrated with Proximal Policy Optimization (PPO).
The algorithm adds mutual information regularization between policy and value function
with adaptive weighting based on environment reward structure.

Key features:
- Information-theoretic regularization I(π_θ; V_φ)
- Adaptive weighting λ(t) based on reward entropy
- Multiple MI estimation methods (MINE, InfoNCE, KDE, histogram)
- Theoretical bounds integration
- Comprehensive logging and analysis

Author: Research team  
Date: May 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import gymnasium as gym
from collections import deque
import math

from rl_neurips.theory.aitpr_bounds import AitprTheory, AitprMutualInformation, compute_reward_entropy


class AitprPolicyNetwork(nn.Module):
    """
    Policy network with feature extraction for mutual information computation.
    
    Supports both discrete and continuous action spaces.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 hidden_dim: int = 256,
                 continuous: bool = False):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.continuous = continuous
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        if continuous:
            # Continuous action space (Gaussian policy)
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std_head = nn.Linear(hidden_dim, action_dim)
        else:
            # Discrete action space (Categorical policy)
            self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action distribution and features.
        
        Returns:
            action_dist: Action distribution (logits or mean/std)
            features: Feature representation for MI computation
        """
        features = self.feature_extractor(state)
        
        if self.continuous:
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            std = torch.exp(log_std.clamp(-20, 2))  # Numerical stability
            return (mean, std), features
        else:
            logits = self.action_head(features)
            return logits, features
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            features: Feature representation
        """
        if self.continuous:
            (mean, std), features = self.forward(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, features
        else:
            logits, features = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, features


class AitprValueNetwork(nn.Module):
    """
    Value network with feature extraction for mutual information computation.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor (same architecture as policy for fair MI computation)
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
        )
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning value estimate and features.
        
        Returns:
            value: State value estimate
            features: Feature representation for MI computation
        """
        features = self.feature_extractor(state)
        value = self.value_head(features)
        return value, features


class AitprPPO:
    """
    AITPR-PPO algorithm implementation.
    
    Integrates information-theoretic regularization with PPO using:
    1. Mutual information estimation between policy and value function
    2. Adaptive regularization weighting based on reward entropy
    3. Theoretical bounds for performance guarantees
    """
    
    def __init__(self,
                 env: gym.Env,
                 policy_lr: float = 3e-4,
                 value_lr: float = 1e-3,
                 hidden_dim: int = 256,
                 lambda_base: float = 1.0,
                 alpha: float = 0.1,
                 mi_method: str = "mine",
                 clip_ratio: float = 0.2,
                 target_kl: float = 0.01,
                 train_pi_iters: int = 80,
                 train_v_iters: int = 80,
                 gamma: float = 0.99,
                 lam: float = 0.97,
                 device: str = "auto"):
        
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.continuous = isinstance(env.action_space, gym.spaces.Box)
        
        if self.continuous:
            self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = env.action_space.n
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Hyperparameters
        self.lambda_base = lambda_base
        self.alpha = alpha
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.gamma = gamma
        self.lam = lam
        
        # Networks
        self.policy = AitprPolicyNetwork(
            self.state_dim, self.action_dim, hidden_dim, self.continuous
        ).to(self.device)
        
        self.value_net = AitprValueNetwork(
            self.state_dim, hidden_dim
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        
        # AITPR components
        self.theory = AitprTheory(gamma=gamma)
        self.mi_estimator = AitprMutualInformation(method=mi_method, hidden_dim=hidden_dim)
        
        # Training state
        self.reward_history = deque(maxlen=1000)
        self.mi_history = deque(maxlen=100)
        self.lambda_history = deque(maxlen=100)
        
        print(f"AITPR-PPO initialized on {self.device}")
        print(f"Action space: {'Continuous' if self.continuous else 'Discrete'}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"MI estimation method: {mi_method}")
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Select action using current policy.
        
        Returns:
            action: Selected action
            log_prob: Log probability of action  
            info: Additional information (features, value, etc.)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, policy_features = self.policy.get_action_and_log_prob(state_tensor)
            value, value_features = self.value_net(state_tensor)
        
        # Convert action to appropriate format for environment
        if self.continuous:
            action_np = action.cpu().numpy().flatten()
        else:
            action_np = action.cpu().numpy().item()  # Scalar for discrete actions
            
        log_prob_np = log_prob.cpu().item()
        
        info = {
            'value': value.cpu().item(),
            'policy_features': policy_features,
            'value_features': value_features,
            'log_prob': log_prob_np
        }
        
        return action_np, log_prob_np, info
    
    def compute_gae(self, 
                   rewards: List[float],
                   values: List[float], 
                   dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        
        return advantages, returns
    
    def compute_adaptive_lambda(self, recent_rewards: List[float]) -> float:
        """
        Compute adaptive regularization weight λ(t).
        
        λ(t) = λ_base * exp(-α * H(R_t))
        """
        if len(recent_rewards) < 10:
            return self.lambda_base
        
        reward_entropy = compute_reward_entropy(np.array(recent_rewards))
        lambda_t = self.theory.adaptive_weight(reward_entropy, self.lambda_base, self.alpha)
        
        self.lambda_history.append(lambda_t)
        return lambda_t
    
    def compute_policy_loss(self,
                           states: torch.Tensor,
                           actions: torch.Tensor, 
                           old_log_probs: torch.Tensor,
                           advantages: torch.Tensor,
                           lambda_t: float) -> Tuple[torch.Tensor, Dict]:
        """
        Compute AITPR policy loss with information-theoretic regularization.
        
        L_AITPR(θ) = L_PPO(θ) + λ(t) * I(π_θ; V_φ)
        """
        # Get current policy and value features
        if self.continuous:
            (mean, std), policy_features = self.policy(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            logits, policy_features = self.policy(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        
        # Get value features
        _, value_features = self.value_net(states)
        
        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Estimate mutual information I(π_θ; V_φ)
        mutual_info = self.mi_estimator.estimate_mutual_info(
            policy_features.detach(),  # Detach to avoid affecting MI network training
            value_features.detach()
        )
        
        # AITPR objective  
        aitpr_loss = policy_loss + lambda_t * mutual_info
        
        # Compute KL divergence for early stopping
        with torch.no_grad():
            kl_div = (old_log_probs - log_probs).mean()
        
        loss_info = {
            'policy_loss': policy_loss.item(),
            'mutual_info': mutual_info,
            'aitpr_loss': aitpr_loss.item(),
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'lambda_t': lambda_t
        }
        
        self.mi_history.append(mutual_info)
        
        return aitpr_loss, loss_info
    
    def compute_value_loss(self,
                          states: torch.Tensor,
                          returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        values, _ = self.value_net(states)
        value_loss = F.mse_loss(values.squeeze(), returns)
        return value_loss
    
    def update(self, batch: Dict) -> Dict:
        """
        Update policy and value function using AITPR.
        
        Args:
            batch: Training batch containing states, actions, rewards, etc.
            
        Returns:
            training_info: Loss and performance metrics
        """
        # Extract batch data
        states = torch.FloatTensor(batch['states']).to(self.device)
        
        # Handle actions based on action space type
        if self.continuous:
            actions = torch.FloatTensor(batch['actions']).to(self.device)
        else:
            # For discrete actions, convert to long tensor and squeeze
            actions = torch.LongTensor(batch['actions']).squeeze(-1).to(self.device)
            
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        rewards = batch['rewards']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute adaptive lambda
        lambda_t = self.compute_adaptive_lambda(rewards)
        
        # Update policy
        policy_info = {'kl_div': 0.0}
        for i in range(self.train_pi_iters):
            self.policy_optimizer.zero_grad()
            aitpr_loss, loss_info = self.compute_policy_loss(
                states, actions, old_log_probs, advantages, lambda_t
            )
            aitpr_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            
            self.policy_optimizer.step()
            
            policy_info = loss_info
            
            # Early stopping based on KL divergence
            if loss_info['kl_div'] > 1.5 * self.target_kl:
                print(f"Early stopping at iteration {i} due to KL divergence")
                break
        
        # Update value function
        value_loss_final = 0.0
        for i in range(self.train_v_iters):
            self.value_optimizer.zero_grad()
            value_loss = self.compute_value_loss(states, returns)
            value_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            
            self.value_optimizer.step()
            value_loss_final = value_loss.item()
        
        # Update reward history
        self.reward_history.extend(rewards)
        
        # Compile training info
        training_info = {
            **policy_info,
            'value_loss': value_loss_final,
            'lambda_t': lambda_t,
            'reward_entropy': compute_reward_entropy(np.array(list(self.reward_history))),
            'avg_reward': np.mean(rewards),
            'avg_return': returns.mean().item()
        }
        
        return training_info
    
    def collect_rollout(self, rollout_length: int = 2048) -> Dict:
        """
        Collect rollout data for training.
        
        Returns:
            batch: Dictionary containing rollout data
        """
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state = self.env.reset()[0]
        
        for step in range(rollout_length):
            # Select action
            action, log_prob, info = self.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store data
            states.append(state)
            # Store actions in consistent format for batch processing
            if self.continuous:
                actions.append(action)  # Already array
            else:
                actions.append([action])  # Make it array-like for consistency
            rewards.append(reward)
            values.append(info['value'])
            log_probs.append(log_prob)
            dones.append(done)
            
            state = next_state
            
            if done:
                state = self.env.reset()[0]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        batch = {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': rewards,
            'values': np.array(values),
            'log_probs': np.array(log_probs),
            'advantages': np.array(advantages),
            'returns': np.array(returns),
            'dones': np.array(dones)
        }
        
        return batch
    
    def train_episode(self, max_steps: int = 2048) -> Dict:
        """
        Train for one episode (rollout + update).
        
        Returns:
            episode_info: Training metrics and performance
        """
        # Collect rollout
        batch = self.collect_rollout(max_steps)
        
        # Update networks
        training_info = self.update(batch)
        
        # Compute theoretical metrics
        policy_features = []
        value_features = []
        states_tensor = torch.FloatTensor(batch['states'][:100]).to(self.device)  # Sample for efficiency
        
        with torch.no_grad():
            for i in range(0, len(states_tensor), 32):  # Process in batches
                batch_states = states_tensor[i:i+32]
                if self.continuous:
                    _, p_features = self.policy(batch_states)
                else:
                    _, p_features = self.policy(batch_states)
                _, v_features = self.value_net(batch_states)
                
                policy_features.append(p_features)
                value_features.append(v_features)
        
        policy_features = torch.cat(policy_features, dim=0)
        value_features = torch.cat(value_features, dim=0)
        
        # Theoretical analysis
        improvement_bound = self.theory.policy_improvement_bound(
            gradient_term=training_info['avg_return'],
            mutual_info=training_info['mutual_info'],
            lambda_t=training_info['lambda_t'],
            theta_norm=1.0  # Placeholder
        )
        
        sample_complexity = self.theory.sample_complexity_bound(
            mutual_info=training_info['mutual_info'],
            epsilon=0.1
        )
        
        episode_info = {
            **training_info,
            'improvement_bound': improvement_bound,
            'sample_complexity': sample_complexity,
            'rollout_length': len(batch['rewards']),
            'total_reward': sum(batch['rewards'])
        }
        
        return episode_info
    
    def save_model(self, filepath: str):
        """Save model parameters."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'hyperparameters': {
                'lambda_base': self.lambda_base,
                'alpha': self.alpha,
                'clip_ratio': self.clip_ratio,
                'target_kl': self.target_kl,
                'gamma': self.gamma,
                'lam': self.lam
            }
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.lambda_base = hyperparams['lambda_base']
        self.alpha = hyperparams['alpha']
        self.clip_ratio = hyperparams['clip_ratio']
        self.target_kl = hyperparams['target_kl']
        self.gamma = hyperparams['gamma']
        self.lam = hyperparams['lam']


if __name__ == "__main__":
    # Example usage and testing
    import gymnasium as gym
    
    # Test environment
    env = gym.make("CartPole-v1")
    
    # Initialize AITPR-PPO
    agent = AitprPPO(
        env=env,
        policy_lr=3e-4,
        value_lr=1e-3,
        lambda_base=1.0,
        alpha=0.1,
        mi_method="mine"
    )
    
    print("AITPR-PPO agent created successfully")
    print(f"Policy network: {sum(p.numel() for p in agent.policy.parameters())} parameters")
    print(f"Value network: {sum(p.numel() for p in agent.value_net.parameters())} parameters")
    
    # Test single episode
    episode_info = agent.train_episode(max_steps=100)
    print("Training episode completed:")
    for key, value in episode_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")