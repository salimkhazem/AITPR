#!/usr/bin/env python3
"""
Comprehensive baseline comparison script for AITPR vs standard RL algorithms.

This script runs AITPR against multiple strong baselines across various environments
to generate publication-quality results for NeurIPS submission.

Usage:
    python scripts/baseline_comparison.py --env all --episodes 100 --seeds 5
    python scripts/baseline_comparison.py --env MountainCarContinuous-v0 --episodes 50 --seeds 3
"""

import argparse
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from rl_neurips.algorithms import (
    AitprPPO, AitprSAC, 
    PPOBaseline, SACBaseline, A2CBaseline,
    create_baseline_agent
)
from rl_neurips.environments import EnvironmentSuite


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging for comparison experiments."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'baseline_comparison.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class BaselineComparison:
    """Comprehensive baseline comparison framework."""
    
    def __init__(self, results_dir: Path = Path("results/baseline_comparison")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.results_dir)
        self.env_suite = EnvironmentSuite()
        
        # Define algorithms to compare
        self.algorithms = {
            'AITPR-PPO-MINE': {'type': 'aitpr_ppo', 'mi_method': 'mine'},
            'AITPR-PPO-InfoNCE': {'type': 'aitpr_ppo', 'mi_method': 'infonce'},
            'PPO-Baseline': {'type': 'ppo'},
            'A2C-Baseline': {'type': 'a2c'},
            'PPO-Entropy': {'type': 'ppo', 'entropy_coef': 0.1},
        }
        
        # Define test environments
        self.environments = [
            # Classic Control - Discrete
            'CartPole-v1',           # Easy discrete control
            'Acrobot-v1',           # Harder discrete control
            'MountainCar-v0',       # Sparse reward discrete
            
            # Classic Control - Continuous  
            'Pendulum-v1',          # Easy continuous control
            'MountainCarContinuous-v0',  # Sparse reward continuous
            'InvertedPendulum-v4',  # Balancing task
            
            # Box2D Environments
            'LunarLander-v3',       # Sparse reward, complex dynamics
            'BipedalWalker-v3',     # Challenging locomotion
            
            # Advanced Control
            'CartPoleSwingUp-v1',   # Harder version of CartPole
        ]
        
    def create_agent(self, algorithm: str, env: gym.Env, config: Dict[str, Any]) -> Any:
        """Create algorithm instance based on configuration."""
        algo_config = self.algorithms[algorithm].copy()
        algo_config.update(config)
        
        if algo_config['type'] == 'aitpr_ppo':
            return AitprPPO(
                env=env,
                mi_method=algo_config.get('mi_method', 'mine'),
                lambda_base=algo_config.get('lambda_base', 1.0),
                alpha=algo_config.get('alpha', 0.1),
                hidden_dim=algo_config.get('hidden_dim', 256),
                gamma=algo_config.get('gamma', 0.99),
                device=algo_config.get('device', 'auto')
            )
        elif algo_config['type'] == 'aitpr_sac':
            return AitprSAC(
                env=env,
                mi_method=algo_config.get('mi_method', 'mine'),
                lambda_base=algo_config.get('lambda_base', 1.0),
                alpha_param=algo_config.get('alpha', 0.1),
                hidden_dim=algo_config.get('hidden_dim', 256),
                device=algo_config.get('device', 'auto')
            )
        else:
            # Handle baseline algorithms with different parameter requirements
            if algo_config['type'] == 'sac':
                return create_baseline_agent(
                    algo_config['type'], 
                    env,
                    hidden_dim=algo_config.get('hidden_dim', 256)
                )
            else:
                return create_baseline_agent(
                    algo_config['type'], 
                    env,
                    entropy_coef=algo_config.get('entropy_coef', 0.01),
                    hidden_dim=algo_config.get('hidden_dim', 256)
                )
    
    def run_single_trial(
        self, 
        algorithm: str, 
        env_name: str, 
        episodes: int = 100,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Run a single trial of algorithm on environment."""
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        
        # Create environment
        env = self.env_suite.make_env(env_name, normalize=True, seed=seed)
        
        # Create agent
        config = {'device': 'auto', 'hidden_dim': 256}
        agent = self.create_agent(algorithm, env, config)
        
        self.logger.info(f"Running {algorithm} on {env_name} (seed={seed})")
        
        # Training metrics
        episode_rewards = []
        training_times = []
        
        start_time = time.time()
        
        # Training loop
        for episode in tqdm(range(episodes), desc=f"{algorithm}-{env_name}", leave=False):
            episode_start = time.time()
            
            # Run episode based on algorithm type
            if hasattr(agent, 'train_episode'):
                # AITPR algorithms
                episode_info = agent.train_episode(max_steps=2048 if 'CartPole' in env_name else 1000)
                episode_reward = episode_info.get('total_reward', episode_info.get('episode_reward', 0))
            else:
                # Baseline algorithms
                episode_info = agent.train_episode(max_steps=2048 if 'CartPole' in env_name else 1000)
                episode_reward = episode_info.get('total_reward', episode_info.get('episode_reward', 0))
            
            episode_rewards.append(episode_reward)
            training_times.append(time.time() - episode_start)
            
            # Log progress every 20 episodes
            if episode % 20 == 0 and episode > 0:
                recent_reward = np.mean(episode_rewards[-20:])
                self.logger.info(f"  Episode {episode}: Avg reward = {recent_reward:.2f}")
        
        training_time = time.time() - start_time
        
        # Evaluation
        eval_rewards = []
        for _ in range(10):
            state = env.reset()[0] if hasattr(env.reset(), '__getitem__') else env.reset()
            total_reward = 0
            done = False
            step_count = 0
            max_steps = 500
            
            while not done and step_count < max_steps:
                if hasattr(agent, 'select_action'):
                    action, _, _ = agent.select_action(state)
                else:
                    action = agent.predict(state)
                    
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                step_count += 1
            
            eval_rewards.append(total_reward)
        
        env.close()
        
        # Compile results
        results = {
            'algorithm': algorithm,
            'environment': env_name,
            'seed': seed,
            'episodes': episodes,
            'training_time': training_time,
            'episode_rewards': episode_rewards,
            'final_reward': np.mean(episode_rewards[-10:]) if episode_rewards else 0,
            'eval_rewards': eval_rewards,
            'eval_mean': np.mean(eval_rewards),
            'eval_std': np.std(eval_rewards),
            'best_reward': np.max(episode_rewards) if episode_rewards else 0,
            'learning_curve': episode_rewards,
            'convergence_episode': self._find_convergence(episode_rewards) if episode_rewards else episodes,
        }
        
        return results
    
    def _find_convergence(self, rewards: List[float], window: int = 20, threshold: float = 0.05) -> int:
        """Find convergence point in learning curve."""
        if len(rewards) < window * 2:
            return len(rewards)
            
        for i in range(window, len(rewards) - window):
            recent_mean = np.mean(rewards[i:i+window])
            future_mean = np.mean(rewards[i+window:i+2*window])
            
            if abs(recent_mean - future_mean) / (abs(recent_mean) + 1e-6) < threshold:
                return i
                
        return len(rewards)
    
    def run_comparison(
        self, 
        environments: List[str] = None,
        algorithms: List[str] = None, 
        episodes: int = 100,
        seeds: int = 3
    ) -> Dict[str, Any]:
        """Run comprehensive comparison across algorithms and environments."""
        
        if environments is None:
            environments = self.environments
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
            
        self.logger.info(f"Starting baseline comparison:")
        self.logger.info(f"  Environments: {environments}")
        self.logger.info(f"  Algorithms: {algorithms}")
        self.logger.info(f"  Episodes per trial: {episodes}")
        self.logger.info(f"  Seeds: {seeds}")
        
        all_results = {}
        
        for env_name in environments:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Environment: {env_name}")
            self.logger.info(f"{'='*60}")
            
            env_results = {}
            
            for algorithm in algorithms:
                self.logger.info(f"\nTesting {algorithm}...")
                
                # Run multiple seeds
                seed_results = []
                for seed in range(seeds):
                    try:
                        result = self.run_single_trial(algorithm, env_name, episodes, seed)
                        seed_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Failed {algorithm} on {env_name} seed {seed}: {e}")
                        continue
                
                if seed_results:
                    # Aggregate results across seeds
                    env_results[algorithm] = self._aggregate_results(seed_results)
                    
                    # Log summary
                    mean_eval = env_results[algorithm]['eval_mean_overall']
                    std_eval = env_results[algorithm]['eval_std_overall']
                    self.logger.info(f"  {algorithm}: {mean_eval:.2f} ± {std_eval:.2f}")
            
            all_results[env_name] = env_results
        
        # Generate summary
        summary = self._generate_summary(all_results)
        
        # Save results
        self._save_results(all_results, summary)
        
        return {
            'detailed_results': all_results,
            'summary': summary
        }
    
    def _aggregate_results(self, seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple seeds."""
        if not seed_results:
            return {}
            
        # Extract eval performance across seeds
        eval_means = [r['eval_mean'] for r in seed_results]
        final_rewards = [r['final_reward'] for r in seed_results]
        convergence_episodes = [r['convergence_episode'] for r in seed_results]
        training_times = [r['training_time'] for r in seed_results]
        
        return {
            'algorithm': seed_results[0]['algorithm'],
            'environment': seed_results[0]['environment'],
            'num_seeds': len(seed_results),
            'episodes': seed_results[0]['episodes'],
            
            # Performance metrics
            'eval_mean_overall': np.mean(eval_means),
            'eval_std_overall': np.std(eval_means),
            'final_reward_mean': np.mean(final_rewards),
            'final_reward_std': np.std(final_rewards),
            
            # Efficiency metrics
            'convergence_mean': np.mean(convergence_episodes),
            'convergence_std': np.std(convergence_episodes),
            'training_time_mean': np.mean(training_times),
            'training_time_std': np.std(training_times),
            
            # Learning curves
            'learning_curves': [r['learning_curve'] for r in seed_results],
            'seed_results': seed_results
        }
    
    def _generate_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics and comparisons."""
        summary = {
            'environments': list(results.keys()),
            'algorithms': [],
            'performance_comparison': {},
            'sample_efficiency': {},
            'statistical_significance': {}
        }
        
        if not results:
            return summary
            
        # Get algorithm list from first environment
        first_env = list(results.keys())[0]
        summary['algorithms'] = list(results[first_env].keys())
        
        # Performance comparison
        for env_name, env_results in results.items():
            summary['performance_comparison'][env_name] = {}
            
            # Find best baseline and AITPR performance
            aitpr_perfs = []
            baseline_perfs = []
            
            for algo, result in env_results.items():
                perf = result.get('eval_mean_overall', 0)
                summary['performance_comparison'][env_name][algo] = perf
                
                if 'AITPR' in algo:
                    aitpr_perfs.append(perf)
                else:
                    baseline_perfs.append(perf)
            
            # Calculate improvement
            if aitpr_perfs and baseline_perfs:
                best_aitpr = max(aitpr_perfs)
                best_baseline = max(baseline_perfs)
                improvement = ((best_aitpr - best_baseline) / (abs(best_baseline) + 1e-6)) * 100
                summary['performance_comparison'][env_name]['improvement_percent'] = improvement
        
        return summary
    
    def _save_results(self, results: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """Save results and generate visualizations."""
        
        # Save detailed results
        results_path = self.results_dir / 'detailed_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        
        # Save summary
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate plots
        self._generate_plots(results)
        
        self.logger.info(f"Results saved to {self.results_dir}")
        self.logger.info(f"  Detailed results: {results_path}")
        self.logger.info(f"  Summary: {summary_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def _generate_plots(self, results: Dict[str, Any]) -> None:
        """Generate comparison plots."""
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Performance comparison bar plot
        self._plot_performance_comparison(results)
        
        # Learning curves
        self._plot_learning_curves(results)
        
        # Sample efficiency comparison
        self._plot_sample_efficiency(results)
    
    def _plot_performance_comparison(self, results: Dict[str, Any]) -> None:
        """Generate performance comparison bar plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (env_name, env_results) in enumerate(results.items()):
            if i >= 4:  # Maximum 4 environments
                break
                
            algorithms = list(env_results.keys())
            means = [env_results[algo].get('eval_mean_overall', 0) for algo in algorithms]
            stds = [env_results[algo].get('eval_std_overall', 0) for algo in algorithms]
            
            # Color AITPR algorithms differently
            colors = ['red' if 'AITPR' in algo else 'blue' for algo in algorithms]
            
            bars = axes[i].bar(range(len(algorithms)), means, yerr=stds, 
                              capsize=5, color=colors, alpha=0.7)
            axes[i].set_title(f'{env_name} Performance', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Evaluation Reward', fontsize=12)
            axes[i].set_xticks(range(len(algorithms)))
            axes[i].set_xticklabels([algo.replace('-', '\n') for algo in algorithms], 
                                   rotation=45, ha='right', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self, results: Dict[str, Any]) -> None:
        """Generate learning curves for each environment."""
        
        for env_name, env_results in results.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for algorithm, result in env_results.items():
                learning_curves = result.get('learning_curves', [])
                if not learning_curves:
                    continue
                    
                # Calculate mean and std across seeds
                max_len = max(len(curve) for curve in learning_curves)
                
                # Pad curves to same length
                padded_curves = []
                for curve in learning_curves:
                    padded = curve + [curve[-1]] * (max_len - len(curve))
                    padded_curves.append(padded)
                
                mean_curve = np.mean(padded_curves, axis=0)
                std_curve = np.std(padded_curves, axis=0)
                episodes = range(len(mean_curve))
                
                # Plot with confidence interval
                color = 'red' if 'AITPR' in algorithm else 'blue'
                line_style = '-' if 'AITPR' in algorithm else '--'
                
                ax.plot(episodes, mean_curve, label=algorithm, 
                       color=color, linestyle=line_style, linewidth=2)
                ax.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                               alpha=0.2, color=color)
            
            ax.set_title(f'Learning Curves: {env_name}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Episode', fontsize=12)
            ax.set_ylabel('Episode Reward', fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'learning_curves_{env_name.replace("-", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_sample_efficiency(self, results: Dict[str, Any]) -> None:
        """Generate sample efficiency comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (env_name, env_results) in enumerate(results.items()):
            if i >= 4:
                break
                
            algorithms = list(env_results.keys())
            convergence_means = [env_results[algo].get('convergence_mean', 0) for algo in algorithms]
            convergence_stds = [env_results[algo].get('convergence_std', 0) for algo in algorithms]
            
            colors = ['red' if 'AITPR' in algo else 'blue' for algo in algorithms]
            
            bars = axes[i].bar(range(len(algorithms)), convergence_means, yerr=convergence_stds,
                              capsize=5, color=colors, alpha=0.7)
            axes[i].set_title(f'{env_name} Sample Efficiency', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Episodes to Convergence', fontsize=12)
            axes[i].set_xticks(range(len(algorithms)))
            axes[i].set_xticklabels([algo.replace('-', '\n') for algo in algorithms], 
                                   rotation=45, ha='right', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean in zip(bars, convergence_means):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(mean)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for baseline comparison."""
    parser = argparse.ArgumentParser(description='Run comprehensive baseline comparison')
    parser.add_argument('--env', type=str, default='all',
                       help='Environment(s) to test (all, or specific env name)')
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                       help='Specific algorithms to test')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Episodes per trial')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds')
    parser.add_argument('--results-dir', type=str, default='results/baseline_comparison',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Initialize comparison framework
    comparison = BaselineComparison(Path(args.results_dir))
    
    # Set environments
    if args.env == 'all':
        environments = comparison.environments
    else:
        environments = [args.env]
    
    # Run comparison
    results = comparison.run_comparison(
        environments=environments,
        algorithms=args.algorithms,
        episodes=args.episodes,
        seeds=args.seeds
    )
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE COMPARISON COMPLETE")
    print("="*80)
    
    for env_name in results['summary']['environments']:
        print(f"\n{env_name}:")
        for algo in results['summary']['algorithms']:
            if algo in results['detailed_results'][env_name]:
                perf = results['detailed_results'][env_name][algo]['eval_mean_overall']
                std = results['detailed_results'][env_name][algo]['eval_std_overall']
                print(f"  {algo:20s}: {perf:8.2f} ± {std:6.2f}")
        
        if 'improvement_percent' in results['summary']['performance_comparison'][env_name]:
            improvement = results['summary']['performance_comparison'][env_name]['improvement_percent']
            print(f"  {'AITPR Improvement:':20s} {improvement:8.1f}%")


if __name__ == "__main__":
    main()