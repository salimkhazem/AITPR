"""
Environment Suite for AITPR Evaluation

This module provides a standardized interface for the 9-environment benchmark
used to evaluate AITPR across diverse RL domains:

Classic Control (3):
- CartPole-v1 (discrete, short episodes)
- Acrobot-v1 (discrete, underactuated control)
- MountainCarContinuous-v0 (continuous, sparse reward)

Continuous Control (4):
- Pendulum-v1 (continuous, dense reward)
- HalfCheetah-v4 (locomotion, high-dimensional)
- Hopper-v4 (locomotion, unstable dynamics)
- Walker2d-v4 (locomotion, bipedal)

Sparse Reward/Exploration (2):
- MiniGrid-DoorKey-6x6-v0 (discrete, procedural)
- MiniGrid-FourRooms-v0 (discrete, navigation)

Features:
- Standardized reward normalization
- Episode length consistency
- Comprehensive evaluation metrics
- Statistical analysis support

Author: Research team
Date: May 2026
"""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from collections import deque
import math

# Handle MiniGrid import
try:
    import gymnasium_minigrid
    MINIGRID_AVAILABLE = True
except ImportError:
    MINIGRID_AVAILABLE = False
    warnings.warn("MiniGrid not available. Some environments will be skipped.")


class NormalizedEnv(gym.Wrapper):
    """
    Environment wrapper for consistent reward normalization and episode handling.
    
    Features:
    - Reward scaling and clipping
    - Observation normalization
    - Episode length standardization
    - Performance tracking
    """
    
    def __init__(self, 
                 env: gym.Env,
                 reward_scale: float = 1.0,
                 reward_clip: float = 10.0,
                 obs_clip: float = 10.0,
                 max_episode_steps: Optional[int] = None):
        super().__init__(env)
        
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.obs_clip = obs_clip
        
        # Override episode length if specified
        if max_episode_steps is not None:
            if hasattr(env, '_max_episode_steps'):
                env._max_episode_steps = max_episode_steps
            self._max_episode_steps = max_episode_steps
        
        # Running statistics for normalization
        self.obs_mean = np.zeros(env.observation_space.shape[0])
        self.obs_var = np.ones(env.observation_space.shape[0])
        self.obs_count = 0
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_steps = 0
        
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        # Update running statistics
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.obs_count
        
        # Normalize
        normalized_obs = (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)
        return np.clip(normalized_obs, -self.obs_clip, self.obs_clip)
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward with scaling and clipping."""
        scaled_reward = reward * self.reward_scale
        return np.clip(scaled_reward, -self.reward_clip, self.reward_clip)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset environment and normalize observation."""
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        return self.normalize_obs(obs), info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step environment with normalization."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Normalize
        normalized_obs = self.normalize_obs(obs)
        normalized_reward = self.normalize_reward(reward)
        
        # Track episode stats
        self.episode_reward += reward  # Track original reward
        self.episode_length += 1
        self.total_steps += 1
        
        # Store episode data when episode ends
        if terminated or truncated:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            
            # Add episode stats to info
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
        
        return normalized_obs, normalized_reward, terminated, truncated, info
    
    def get_statistics(self) -> Dict:
        """Get environment performance statistics."""
        if len(self.episode_rewards) == 0:
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_length': 0,
                'total_episodes': 0,
                'total_steps': self.total_steps
            }
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'total_episodes': len(self.episode_rewards),
            'total_steps': self.total_steps,
            'recent_mean_reward': np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
        }


class EnvironmentSuite:
    """
    Complete environment suite for AITPR evaluation.
    
    Provides standardized access to all 9 benchmark environments
    with consistent interfaces and evaluation protocols.
    """
    
    def __init__(self):
        self.environments = self._get_environment_configs()
        self.available_envs = self._check_environment_availability()
        
    def _get_environment_configs(self) -> Dict[str, Dict]:
        """Define configuration for all benchmark environments."""
        configs = {
            # Classic Control
            'CartPole-v1': {
                'env_id': 'CartPole-v1',
                'category': 'classic_control',
                'action_space': 'discrete',
                'reward_type': 'dense',
                'difficulty': 'easy',
                'max_steps': 500,
                'reward_scale': 1.0,
                'target_reward': 475,  # Success threshold
                'description': 'Balance pole on cart, discrete actions, termination conditions'
            },
            'Acrobot-v1': {
                'env_id': 'Acrobot-v1',
                'category': 'classic_control',
                'action_space': 'discrete', 
                'reward_type': 'sparse',
                'difficulty': 'medium',
                'max_steps': 500,
                'reward_scale': 1.0,
                'target_reward': -100,  # Success threshold (negative reward, higher is better)
                'description': 'Swing acrobot up using underactuated control'
            },
            'MountainCarContinuous-v0': {
                'env_id': 'MountainCarContinuous-v0',
                'category': 'classic_control',
                'action_space': 'continuous',
                'reward_type': 'sparse',
                'difficulty': 'medium',
                'max_steps': 999,
                'reward_scale': 1.0,
                'target_reward': 90,  # Success threshold
                'description': 'Drive car up mountain with continuous force control'
            },
            
            # Continuous Control
            'Pendulum-v1': {
                'env_id': 'Pendulum-v1',
                'category': 'continuous_control',
                'action_space': 'continuous',
                'reward_type': 'dense',
                'difficulty': 'easy',
                'max_steps': 200,
                'reward_scale': 1.0,
                'target_reward': -150,  # Success threshold (negative reward)
                'description': 'Inverted pendulum stabilization with torque control'
            },
            'HalfCheetah-v4': {
                'env_id': 'HalfCheetah-v4',
                'category': 'continuous_control',
                'action_space': 'continuous',
                'reward_type': 'dense',
                'difficulty': 'hard',
                'max_steps': 1000,
                'reward_scale': 0.1,  # Scale down for normalization
                'target_reward': 4000,  # Success threshold
                'description': 'Half-cheetah locomotion, high-dimensional continuous control'
            },
            'Hopper-v4': {
                'env_id': 'Hopper-v4',
                'category': 'continuous_control',
                'action_space': 'continuous',
                'reward_type': 'dense',
                'difficulty': 'hard',
                'max_steps': 1000,
                'reward_scale': 0.1,
                'target_reward': 3000,  # Success threshold
                'description': 'Hopper locomotion with unstable dynamics'
            },
            'Walker2d-v4': {
                'env_id': 'Walker2d-v4',
                'category': 'continuous_control',
                'action_space': 'continuous',
                'reward_type': 'dense',
                'difficulty': 'hard',
                'max_steps': 1000,
                'reward_scale': 0.1,
                'target_reward': 4000,  # Success threshold
                'description': 'Bipedal walker locomotion'
            },
            
            # Sparse Reward / Exploration
            'MiniGrid-DoorKey-6x6-v0': {
                'env_id': 'MiniGrid-DoorKey-6x6-v0',
                'category': 'sparse_reward',
                'action_space': 'discrete',
                'reward_type': 'sparse',
                'difficulty': 'medium',
                'max_steps': 100,
                'reward_scale': 1.0,
                'target_reward': 0.9,  # Success threshold (success rate)
                'description': 'Navigate grid world, pick up key, unlock door'
            },
            'MiniGrid-FourRooms-v0': {
                'env_id': 'MiniGrid-FourRooms-v0',
                'category': 'sparse_reward',
                'action_space': 'discrete',
                'reward_type': 'sparse', 
                'difficulty': 'medium',
                'max_steps': 100,
                'reward_scale': 1.0,
                'target_reward': 0.9,  # Success threshold (success rate)
                'description': 'Navigate four-room grid world to reach goal'
            }
        }
        
        return configs
    
    def _check_environment_availability(self) -> List[str]:
        """Check which environments are available."""
        available = []
        
        for env_name, config in self.environments.items():
            try:
                if env_name.startswith('MiniGrid') and not MINIGRID_AVAILABLE:
                    print(f"Skipping {env_name}: MiniGrid not available")
                    continue
                
                # Try to create environment
                env = gym.make(config['env_id'])
                env.close()
                available.append(env_name)
                print(f"✓ {env_name} available")
                
            except Exception as e:
                print(f"✗ {env_name} not available: {e}")
        
        return available
    
    def make_env(self, 
                 env_name: str, 
                 normalize: bool = True,
                 seed: Optional[int] = None) -> gym.Env:
        """
        Create environment instance with optional normalization.
        
        Args:
            env_name: Name of environment from suite
            normalize: Whether to apply normalization wrapper
            seed: Random seed
            
        Returns:
            Environment instance
        """
        if env_name not in self.available_envs:
            raise ValueError(f"Environment {env_name} not available. Available: {self.available_envs}")
        
        config = self.environments[env_name]
        
        # Create base environment
        env = gym.make(config['env_id'])
        
        # Set seed if provided
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        
        # Apply normalization wrapper
        if normalize:
            env = NormalizedEnv(
                env,
                reward_scale=config['reward_scale'],
                max_episode_steps=config['max_steps']
            )
        
        return env
    
    def get_env_info(self, env_name: str) -> Dict:
        """Get detailed information about environment."""
        if env_name not in self.environments:
            raise ValueError(f"Unknown environment: {env_name}")
        
        config = self.environments[env_name]
        
        # Create temporary environment to get space information
        env = gym.make(config['env_id'])
        
        info = {
            **config,
            'state_dim': env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None,
            'action_dim': (env.action_space.shape[0] if hasattr(env.action_space, 'shape') 
                          else env.action_space.n),
            'action_low': (env.action_space.low if hasattr(env.action_space, 'low') else None),
            'action_high': (env.action_space.high if hasattr(env.action_space, 'high') else None),
            'obs_space_type': type(env.observation_space).__name__,
            'action_space_type': type(env.action_space).__name__,
            'available': env_name in self.available_envs
        }
        
        env.close()
        return info
    
    def get_category_envs(self, category: str) -> List[str]:
        """Get all environments in a specific category."""
        category_envs = []
        for env_name, config in self.environments.items():
            if config['category'] == category and env_name in self.available_envs:
                category_envs.append(env_name)
        return category_envs
    
    def get_all_available_envs(self) -> List[str]:
        """Get list of all available environments."""
        return self.available_envs.copy()
    
    def run_environment_test(self, env_name: str, episodes: int = 5) -> Dict:
        """
        Run test episodes to verify environment functionality.
        
        Args:
            env_name: Environment to test
            episodes: Number of test episodes
            
        Returns:
            Test results and statistics
        """
        env = self.make_env(env_name, normalize=True, seed=42)
        
        episode_rewards = []
        episode_lengths = []
        errors = []
        
        for episode in range(episodes):
            try:
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                
                while True:
                    # Random action
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
            except Exception as e:
                errors.append(str(e))
        
        env.close()
        
        test_results = {
            'env_name': env_name,
            'episodes_completed': len(episode_rewards),
            'episodes_failed': len(errors),
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'std_length': np.std(episode_lengths) if episode_lengths else 0,
            'errors': errors[:3],  # First 3 errors
            'success': len(errors) == 0
        }
        
        return test_results
    
    def print_suite_summary(self):
        """Print comprehensive summary of environment suite."""
        print("\n" + "="*80)
        print("AITPR ENVIRONMENT SUITE SUMMARY")
        print("="*80)
        
        print(f"\nTotal environments: {len(self.environments)}")
        print(f"Available environments: {len(self.available_envs)}")
        
        # Group by category
        categories = {}
        for env_name in self.available_envs:
            category = self.environments[env_name]['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(env_name)
        
        for category, envs in categories.items():
            print(f"\n{category.upper().replace('_', ' ')} ({len(envs)} environments):")
            for env in envs:
                info = self.get_env_info(env)
                print(f"  ✓ {env}")
                print(f"    - Action space: {info['action_space']} ({info['action_dim']} dim)")
                print(f"    - Reward type: {info['reward_type']}")
                print(f"    - Difficulty: {info['difficulty']}")
                print(f"    - Target: {info['target_reward']}")
        
        print("\n" + "="*80)


def create_evaluation_environments(seed_base: int = 0) -> Dict[str, gym.Env]:
    """
    Create all evaluation environments with different seeds.
    
    Args:
        seed_base: Base seed for reproducibility
        
    Returns:
        Dictionary mapping env names to environment instances
    """
    suite = EnvironmentSuite()
    environments = {}
    
    for i, env_name in enumerate(suite.get_all_available_envs()):
        env_seed = seed_base + i
        env = suite.make_env(env_name, normalize=True, seed=env_seed)
        environments[env_name] = env
        print(f"Created {env_name} with seed {env_seed}")
    
    return environments


def run_environment_suite_test():
    """Run comprehensive test of all environments in the suite."""
    suite = EnvironmentSuite()
    
    print("Testing all environments...")
    all_results = {}
    
    for env_name in suite.get_all_available_envs():
        print(f"\nTesting {env_name}...")
        results = suite.run_environment_test(env_name, episodes=3)
        all_results[env_name] = results
        
        if results['success']:
            print(f"  ✓ Success - Mean reward: {results['mean_reward']:.2f}, Mean length: {results['mean_length']:.1f}")
        else:
            print(f"  ✗ Failed - {len(results['errors'])} errors")
            for error in results['errors']:
                print(f"    Error: {error}")
    
    # Summary
    successful = sum(1 for r in all_results.values() if r['success'])
    total = len(all_results)
    print(f"\nTest Summary: {successful}/{total} environments working correctly")
    
    return all_results


if __name__ == "__main__":
    # Initialize and test environment suite
    suite = EnvironmentSuite()
    suite.print_suite_summary()
    
    # Run comprehensive test
    test_results = run_environment_suite_test()
    
    # Example usage
    print("\nExample: Creating CartPole environment")
    env = suite.make_env('CartPole-v1', normalize=True, seed=42)
    obs, _ = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    env.close()