"""
AITPR-SAC: Adaptive Information-Theoretic Policy Regularization with SAC

This module implements the AITPR method integrated with Soft Actor-Critic (SAC).
The algorithm adds mutual information regularization between policy and critic networks
with adaptive weighting based on environment reward structure.

Key features:
- Information-theoretic regularization I(π_θ; Q_φ)
- Adaptive weighting λ(t) based on reward entropy  
- Twin critic networks with feature extraction
- Automatic temperature tuning α
- Comprehensive theoretical integration

Author: Research team
Date: May 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from collections import deque
import math

from ..theory.aitpr_bounds import AitprTheory, AitprMutualInformation, compute_reward_entropy


class AitprSACPolicyNetwork(nn.Module):
    """
    Policy network for AITPR-SAC with feature extraction.
    
    Implements continuous policy with tanh squashing and reparameterization trick.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Policy heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean, log_std, and features.
        
        Returns:
            mean: Action mean (before tanh)
            log_std: Log standard deviation (clamped)
            features: Feature representation for MI computation
        """
        features = self.feature_extractor(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, features
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action using reparameterization trick.
        
        Returns:
            action: Sampled action (after tanh)
            log_prob: Log probability of action
            features: Feature representation
        """
        mean, log_std, features = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        action = torch.tanh(x_t)
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, features
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given action.
        """
        mean, log_std, _ = self.forward(state)
        std = log_std.exp()
        
        # Inverse tanh to get pre-activation
        action_clamped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        x_t = torch.atanh(action_clamped)
        
        # Log probability with correction
        normal = Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return log_prob


class AitprSACCriticNetwork(nn.Module):
    """
    Critic network (Q-function) with feature extraction for MI computation.
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Q-value head
        self.q_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning Q-value and features.
        
        Returns:
            q_value: Q(s,a) estimate
            features: Feature representation for MI computation
        """
        x = torch.cat([state, action], dim=1)
        features = self.feature_extractor(x)
        q_value = self.q_head(features)
        
        return q_value, features


class ReplayBuffer:
    """
    Replay buffer for off-policy learning.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        # Allocate memory
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros((capacity, 1))
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add transition to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }
    
    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size


class AitprSAC:
    """
    AITPR-SAC algorithm implementation.
    
    Integrates information-theoretic regularization with SAC using:
    1. Mutual information between policy and critic features
    2. Adaptive regularization based on reward entropy  
    3. Twin critic networks for reduced overestimation
    4. Automatic temperature tuning
    """
    
    def __init__(self,
                 env: gym.Env,
                 policy_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 hidden_dim: int = 256,
                 lambda_base: float = 1.0,
                 alpha_param: float = 0.1,
                 mi_method: str = "mine",
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 automatic_entropy_tuning: bool = True,
                 buffer_size: int = 1000000,
                 batch_size: int = 256,
                 device: str = "auto"):
        
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Hyperparameters
        self.lambda_base = lambda_base
        self.alpha_param = alpha_param  # For adaptive weighting (different from entropy coefficient)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Networks
        self.policy = AitprSACPolicyNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.device)
        
        # Twin critics
        self.critic1 = AitprSACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.device)
        
        self.critic2 = AitprSACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.device)
        
        # Target critics
        self.critic1_target = AitprSACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.device)
        
        self.critic2_target = AitprSACCriticNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.device)
        
        # Initialize targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Entropy temperature
        if automatic_entropy_tuning:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim)
        
        # AITPR components
        self.theory = AitprTheory(gamma=gamma)
        self.mi_estimator = AitprMutualInformation(method=mi_method, hidden_dim=hidden_dim)
        
        # Training state
        self.reward_history = deque(maxlen=1000)
        self.mi_history = deque(maxlen=100)
        self.lambda_history = deque(maxlen=100)
        self.total_steps = 0
        
        print(f"AITPR-SAC initialized on {self.device}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        print(f"MI estimation method: {mi_method}")
        print(f"Automatic entropy tuning: {automatic_entropy_tuning}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Select action using current policy.
        
        Args:
            state: Current state
            deterministic: Use mean action (for evaluation)
            
        Returns:
            action: Selected action
            info: Additional information
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _, policy_features = self.policy(state_tensor)
                action = torch.tanh(mean)
                log_prob = torch.zeros(1)
            else:
                action, log_prob, policy_features = self.policy.sample(state_tensor)
        
        action_np = action.cpu().numpy().flatten()
        
        info = {
            'log_prob': log_prob.cpu().item(),
            'policy_features': policy_features
        }
        
        return action_np, info
    
    def compute_adaptive_lambda(self, recent_rewards: List[float]) -> float:
        """
        Compute adaptive regularization weight λ(t).
        
        λ(t) = λ_base * exp(-α * H(R_t))
        """
        if len(recent_rewards) < 10:
            return self.lambda_base
        
        reward_entropy = compute_reward_entropy(np.array(recent_rewards))
        lambda_t = self.theory.adaptive_weight(reward_entropy, self.lambda_base, self.alpha_param)
        
        self.lambda_history.append(lambda_t)
        return lambda_t
    
    def update_critic(self, batch: Dict, lambda_t: float) -> Dict:
        """
        Update critic networks with AITPR regularization.
        """
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            
            # Target Q-values using twin critics
            q1_target, _ = self.critic1_target(next_states, next_actions)
            q2_target, _ = self.critic2_target(next_states, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            # SAC target with entropy bonus
            if self.automatic_entropy_tuning:
                alpha_value = self.alpha
            else:
                alpha_value = self.alpha
            
            target_value = rewards + self.gamma * (1 - dones) * (min_q_target - alpha_value * next_log_probs)
        
        # Current Q-values and features
        q1_current, q1_features = self.critic1(states, actions)
        q2_current, q2_features = self.critic2(states, actions)
        
        # Get policy features for MI computation
        _, _, policy_features = self.policy(states)
        
        # Standard critic losses
        critic1_loss = F.mse_loss(q1_current, target_value)
        critic2_loss = F.mse_loss(q2_current, target_value)
        
        # Estimate mutual information between policy and critics
        mi_policy_q1 = self.mi_estimator.estimate_mutual_info(
            policy_features.detach(), q1_features.detach()
        )
        mi_policy_q2 = self.mi_estimator.estimate_mutual_info(
            policy_features.detach(), q2_features.detach()
        )
        
        avg_mutual_info = (mi_policy_q1 + mi_policy_q2) / 2.0
        
        # AITPR critic losses
        aitpr_critic1_loss = critic1_loss + lambda_t * mi_policy_q1
        aitpr_critic2_loss = critic2_loss + lambda_t * mi_policy_q2
        
        # Update critic 1
        self.critic1_optimizer.zero_grad()
        aitpr_critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        # Update critic 2
        self.critic2_optimizer.zero_grad()
        aitpr_critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        self.mi_history.append(avg_mutual_info)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'aitpr_critic1_loss': aitpr_critic1_loss.item(),
            'aitpr_critic2_loss': aitpr_critic2_loss.item(),
            'mutual_info': avg_mutual_info,
            'q1_mean': q1_current.mean().item(),
            'q2_mean': q2_current.mean().item()
        }
    
    def update_policy_and_alpha(self, batch: Dict) -> Dict:
        """
        Update policy and entropy coefficient.
        """
        states = batch['states'].to(self.device)
        
        # Sample actions from current policy
        new_actions, log_probs, policy_features = self.policy.sample(states)
        
        # Q-values for new actions
        q1_new, _ = self.critic1(states, new_actions)
        q2_new, _ = self.critic2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        
        # Policy loss (SAC objective)
        if self.automatic_entropy_tuning:
            alpha_value = self.alpha
        else:
            alpha_value = self.alpha
        
        policy_loss = (alpha_value * log_probs - min_q_new).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # Update entropy coefficient
        alpha_loss = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        return {
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else alpha_loss,
            'alpha_value': alpha_value.item() if isinstance(alpha_value, torch.Tensor) else alpha_value,
            'entropy': -log_probs.mean().item()
        }
    
    def update_target_networks(self):
        """Soft update target networks."""
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def update(self) -> Dict:
        """
        Update all networks using AITPR.
        
        Returns:
            training_info: Loss and performance metrics
        """
        if not self.replay_buffer.ready(self.batch_size):
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Extract recent rewards for adaptive lambda
        recent_rewards = list(self.reward_history)[-100:]  # Last 100 rewards
        lambda_t = self.compute_adaptive_lambda(recent_rewards)
        
        # Update critics with AITPR
        critic_info = self.update_critic(batch, lambda_t)
        
        # Update policy and alpha
        policy_info = self.update_policy_and_alpha(batch)
        
        # Update target networks
        self.update_target_networks()
        
        # Compile training info
        training_info = {
            **critic_info,
            **policy_info,
            'lambda_t': lambda_t,
            'reward_entropy': compute_reward_entropy(np.array(recent_rewards)) if recent_rewards else 0.0
        }
        
        return training_info
    
    def train_episode(self, max_steps: int = 1000) -> Dict:
        """
        Train for one episode.
        
        Returns:
            episode_info: Training metrics and performance
        """
        state = self.env.reset()[0]
        episode_reward = 0
        episode_steps = 0
        training_infos = []
        
        for step in range(max_steps):
            # Select action
            action, _ = self.select_action(state)
            
            # Environment step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, done)
            self.reward_history.append(reward)
            
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # Update networks
            if self.total_steps > self.batch_size:
                training_info = self.update()
                if training_info:
                    training_infos.append(training_info)
            
            state = next_state
            
            if done:
                break
        
        # Aggregate training info
        if training_infos:
            episode_training_info = {}
            for key in training_infos[0].keys():
                episode_training_info[key] = np.mean([info[key] for info in training_infos])
        else:
            episode_training_info = {}
        
        # Compute theoretical metrics
        if len(self.mi_history) > 0:
            recent_mi = np.mean(list(self.mi_history)[-10:])
            
            improvement_bound = self.theory.policy_improvement_bound(
                gradient_term=episode_reward,
                mutual_info=recent_mi,
                lambda_t=episode_training_info.get('lambda_t', 1.0),
                theta_norm=1.0  # Placeholder
            )
            
            sample_complexity = self.theory.sample_complexity_bound(
                mutual_info=recent_mi,
                epsilon=0.1
            )
        else:
            improvement_bound = 0.0
            sample_complexity = 0
        
        episode_info = {
            **episode_training_info,
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'total_steps': self.total_steps,
            'improvement_bound': improvement_bound,
            'sample_complexity': sample_complexity,
            'buffer_size': self.replay_buffer.size
        }
        
        return episode_info
    
    def save_model(self, filepath: str):
        """Save model parameters."""
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'hyperparameters': {
                'lambda_base': self.lambda_base,
                'alpha_param': self.alpha_param,
                'gamma': self.gamma,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'automatic_entropy_tuning': self.automatic_entropy_tuning
            },
            'total_steps': self.total_steps
        }
        
        if self.automatic_entropy_tuning:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.lambda_base = hyperparams['lambda_base']
        self.alpha_param = hyperparams['alpha_param']
        self.gamma = hyperparams['gamma']
        self.tau = hyperparams['tau']
        self.batch_size = hyperparams['batch_size']
        self.automatic_entropy_tuning = hyperparams['automatic_entropy_tuning']
        
        self.total_steps = checkpoint.get('total_steps', 0)


if __name__ == "__main__":
    # Example usage and testing
    import gymnasium as gym
    
    # Test environment
    env = gym.make("Pendulum-v1")
    
    # Initialize AITPR-SAC
    agent = AitprSAC(
        env=env,
        policy_lr=3e-4,
        critic_lr=3e-4,
        lambda_base=1.0,
        alpha_param=0.1,
        mi_method="mine",
        automatic_entropy_tuning=True
    )
    
    print("AITPR-SAC agent created successfully")
    print(f"Policy network: {sum(p.numel() for p in agent.policy.parameters())} parameters")
    print(f"Critic networks: {sum(p.numel() for p in agent.critic1.parameters())} parameters each")
    
    # Test single episode
    episode_info = agent.train_episode(max_steps=100)
    print("Training episode completed:")
    for key, value in episode_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")