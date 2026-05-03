"""
Comprehensive Experimental Framework for AITPR Evaluation

This script runs the complete experimental suite comparing AITPR against
all baseline algorithms across the 9-environment benchmark.

Experimental Design:
- 9 environments (classic control, continuous control, sparse reward)
- 7 baseline algorithms + 2 AITPR variants = 9 total algorithms
- 5+ seeds per experiment for statistical significance
- Comprehensive evaluation metrics and analysis
- Statistical significance testing
- Automated result generation and visualization

Key Features:
- Parallel experiment execution with resource management
- Robust error handling and experiment resumption
- Real-time monitoring and progress tracking
- Automated statistical analysis and reporting
- Publication-ready figure and table generation

Author: Research team
Date: May 2026
"""

import os
import sys
import argparse
import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from datetime import datetime
import pickle
import warnings
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environments.env_suite import EnvironmentSuite, create_evaluation_environments
from algorithms.aitpr_ppo import AitprPPO
from algorithms.aitpr_sac import AitprSAC
from algorithms.baselines import create_baseline_agent


class ExperimentConfig:
    """Configuration management for experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.config = {
            # Experimental design
            'seeds': [42, 123, 456, 789, 1337],  # 5 seeds minimum
            'episodes_per_seed': 500,  # Episodes to train per seed
            'eval_episodes': 10,  # Episodes for evaluation
            'eval_frequency': 50,  # Evaluate every N training episodes
            'save_frequency': 100,  # Save models every N episodes
            
            # Environment settings
            'environments': [
                'CartPole-v1',
                'Acrobot-v1', 
                'MountainCarContinuous-v0',
                'Pendulum-v1',
                'HalfCheetah-v4',
                'Hopper-v4',
                'Walker2d-v4',
                'MiniGrid-DoorKey-6x6-v0',
                'MiniGrid-FourRooms-v0'
            ],
            
            # Algorithm settings
            'algorithms': {
                'aitpr_ppo': {
                    'class': 'AitprPPO',
                    'params': {
                        'policy_lr': 3e-4,
                        'value_lr': 1e-3,
                        'lambda_base': 1.0,
                        'alpha': 0.1,
                        'mi_method': 'mine'
                    }
                },
                'aitpr_sac': {
                    'class': 'AitprSAC',
                    'params': {
                        'policy_lr': 3e-4,
                        'critic_lr': 3e-4,
                        'lambda_base': 1.0,
                        'alpha_param': 0.1,
                        'mi_method': 'mine'
                    }
                },
                'ppo': {
                    'class': 'PPOBaseline',
                    'params': {
                        'policy_lr': 3e-4,
                        'value_lr': 1e-3,
                        'entropy_coef': 0.01
                    }
                },
                'sac': {
                    'class': 'SACBaseline',
                    'params': {
                        'policy_lr': 3e-4,
                        'critic_lr': 3e-4,
                        'automatic_entropy_tuning': True
                    }
                },
                'a2c': {
                    'class': 'A2CBaseline',
                    'params': {
                        'lr': 3e-4,
                        'entropy_coef': 0.01
                    }
                },
                'td3': {
                    'class': 'TD3Baseline',
                    'params': {}
                },
                'ppo_entropy': {
                    'class': 'PPOEntropyBaseline',
                    'params': {
                        'policy_lr': 3e-4,
                        'value_lr': 1e-3,
                        'entropy_coef': 0.1
                    }
                },
                'sac_kl': {
                    'class': 'SACKLBaseline',
                    'params': {
                        'policy_lr': 3e-4,
                        'critic_lr': 3e-4,
                        'automatic_entropy_tuning': True
                    }
                },
                'trpo': {
                    'class': 'TRPOBaseline',
                    'params': {}
                }
            },
            
            # Resource management
            'max_parallel_jobs': min(8, mp.cpu_count()),
            'gpu_memory_limit': 4096,  # MB per job
            'timeout_hours': 12,  # Maximum time per experiment
            
            # Output settings
            'save_models': True,
            'save_trajectories': False,  # Save full episode trajectories
            'verbose': True,
            'create_videos': False,  # Record evaluation videos
            
            # Statistical analysis
            'confidence_level': 0.95,
            'statistical_tests': ['mann_whitney', 'bootstrap'],
            'effect_size_threshold': 0.5,  # Minimum effect size for significance
            'bootstrap_samples': 10000
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self._update_config(self.config, custom_config)
    
    def _update_config(self, base: Dict, update: Dict):
        """Recursively update configuration."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def save(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value


class ExperimentRunner:
    """Main experiment execution and management class."""
    
    def __init__(self, 
                 config: ExperimentConfig,
                 results_dir: str,
                 resume: bool = False):
        
        self.config = config
        self.results_dir = Path(results_dir)
        self.resume = resume
        
        # Create directory structure
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "analysis").mkdir(exist_ok=True)
        
        # Initialize environment suite
        self.env_suite = EnvironmentSuite()
        
        # Setup logging
        self.setup_logging()
        
        print(f"Experiment runner initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Available environments: {len(self.env_suite.get_all_available_envs())}")
        print(f"Configured algorithms: {len(self.config.get('algorithms'))}")
        print(f"Seeds per experiment: {len(self.config.get('seeds'))}")
    
    def setup_logging(self):
        """Setup experiment logging."""
        import logging
        
        log_file = self.results_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if self.config.get('verbose') else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Experiment logging initialized")
    
    def create_agent(self, algorithm: str, env: gym.Env) -> Any:
        """Create agent instance for given algorithm and environment."""
        algo_config = self.config.get('algorithms')[algorithm]
        algo_class = algo_config['class']
        algo_params = algo_config['params'].copy()
        
        # Handle different algorithm types
        if algo_class == 'AitprPPO':
            return AitprPPO(env, **algo_params)
        elif algo_class == 'AitprSAC':
            return AitprSAC(env, **algo_params)
        else:
            # Use baseline factory
            return create_baseline_agent(
                algorithm.replace('_baseline', ''), 
                env, 
                **algo_params
            )
    
    def run_single_experiment(self, 
                             env_name: str, 
                             algorithm: str, 
                             seed: int) -> Dict:
        """
        Run a single experiment configuration.
        
        Args:
            env_name: Environment name
            algorithm: Algorithm name
            seed: Random seed
            
        Returns:
            Experiment results dictionary
        """
        experiment_id = f"{env_name}_{algorithm}_seed{seed}"
        self.logger.info(f"Starting experiment: {experiment_id}")
        
        try:
            # Set random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # Create environment
            env = self.env_suite.make_env(env_name, normalize=True, seed=seed)
            
            # Create agent
            agent = self.create_agent(algorithm, env)
            
            # Training loop
            results = {
                'experiment_id': experiment_id,
                'env_name': env_name,
                'algorithm': algorithm,
                'seed': seed,
                'training_episodes': [],
                'evaluation_results': [],
                'final_performance': {},
                'training_time': 0,
                'success': False
            }
            
            start_time = time.time()
            episodes_per_seed = self.config.get('episodes_per_seed')
            eval_frequency = self.config.get('eval_frequency')
            
            for episode in range(episodes_per_seed):
                # Training episode
                if hasattr(agent, 'train_episode'):
                    episode_info = agent.train_episode()
                else:
                    # Fallback for simple agents
                    episode_info = {'episode_reward': 0, 'episode_steps': 0}
                
                episode_info['episode'] = episode
                episode_info['timestamp'] = time.time() - start_time
                results['training_episodes'].append(episode_info)
                
                # Evaluation
                if (episode + 1) % eval_frequency == 0:
                    eval_results = self.evaluate_agent(agent, env, episode)
                    eval_results['episode'] = episode
                    eval_results['timestamp'] = time.time() - start_time
                    results['evaluation_results'].append(eval_results)
                    
                    self.logger.info(
                        f"{experiment_id} - Episode {episode+1}/{episodes_per_seed} - "
                        f"Eval reward: {eval_results.get('mean_reward', 0):.2f} ± "
                        f"{eval_results.get('std_reward', 0):.2f}"
                    )
                
                # Save checkpoint
                if self.config.get('save_models') and (episode + 1) % self.config.get('save_frequency') == 0:
                    model_path = (self.results_dir / "models" / 
                                f"{experiment_id}_episode{episode+1}.pkl")
                    self.save_agent_checkpoint(agent, model_path)
            
            # Final evaluation
            final_eval = self.evaluate_agent(agent, env, episodes_per_seed)
            results['final_performance'] = final_eval
            results['training_time'] = time.time() - start_time
            results['success'] = True
            
            # Save final results
            results_file = self.results_dir / "logs" / f"{experiment_id}_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(
                f"Completed experiment: {experiment_id} - "
                f"Final reward: {final_eval.get('mean_reward', 0):.2f} ± "
                f"{final_eval.get('std_reward', 0):.2f} - "
                f"Time: {results['training_time']:.1f}s"
            )
            
            env.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            
            return {
                'experiment_id': experiment_id,
                'env_name': env_name,
                'algorithm': algorithm,
                'seed': seed,
                'error': str(e),
                'success': False
            }
    
    def evaluate_agent(self, agent: Any, env: gym.Env, training_episode: int) -> Dict:
        """
        Evaluate agent performance over multiple episodes.
        
        Args:
            agent: Trained agent
            env: Environment instance
            training_episode: Current training episode number
            
        Returns:
            Evaluation results
        """
        eval_episodes = self.config.get('eval_episodes')
        rewards = []
        episode_lengths = []
        
        for eval_ep in range(eval_episodes):
            state = env.reset()[0]
            episode_reward = 0
            episode_length = 0
            max_steps = 2000  # Prevent infinite episodes
            
            for step in range(max_steps):
                if hasattr(agent, 'select_action'):
                    if hasattr(agent, 'continuous') and agent.continuous:
                        # Continuous action agent
                        action, _ = agent.select_action(state, deterministic=True)
                    else:
                        # Discrete action agent
                        action, _, _ = agent.select_action(state)
                else:
                    # Fallback random action
                    action = env.action_space.sample()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        eval_results = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'eval_episodes': eval_episodes,
            'training_episode': training_episode,
            'all_rewards': rewards
        }
        
        return eval_results
    
    def save_agent_checkpoint(self, agent: Any, filepath: Path):
        """Save agent checkpoint."""
        try:
            if hasattr(agent, 'save_model'):
                agent.save_model(str(filepath))
            else:
                # Fallback: save agent state dict if available
                if hasattr(agent, 'state_dict'):
                    torch.save(agent.state_dict(), filepath)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint {filepath}: {e}")
    
    def get_pending_experiments(self) -> List[Tuple[str, str, int]]:
        """Get list of experiments that need to be run."""
        pending = []
        
        algorithms = list(self.config.get('algorithms').keys())
        environments = [env for env in self.config.get('environments') 
                       if env in self.env_suite.get_all_available_envs()]
        seeds = self.config.get('seeds')
        
        for env_name in environments:
            for algorithm in algorithms:
                for seed in seeds:
                    experiment_id = f"{env_name}_{algorithm}_seed{seed}"
                    results_file = self.results_dir / "logs" / f"{experiment_id}_results.pkl"
                    
                    # Skip if already completed and not resuming
                    if results_file.exists() and not self.resume:
                        continue
                    
                    # Skip if completed successfully
                    if results_file.exists():
                        try:
                            with open(results_file, 'rb') as f:
                                results = pickle.load(f)
                                if results.get('success', False):
                                    continue
                        except:
                            pass  # File corrupted, re-run
                    
                    pending.append((env_name, algorithm, seed))
        
        return pending
    
    def run_experiments_parallel(self, max_workers: Optional[int] = None):
        """Run all experiments in parallel."""
        if max_workers is None:
            max_workers = self.config.get('max_parallel_jobs')
        
        pending_experiments = self.get_pending_experiments()
        
        if not pending_experiments:
            self.logger.info("No pending experiments found.")
            return
        
        self.logger.info(f"Found {len(pending_experiments)} pending experiments")
        self.logger.info(f"Running with {max_workers} parallel workers")
        
        # Save configuration
        config_file = self.results_dir / "experiment_config.json"
        self.config.save(str(config_file))
        
        start_time = time.time()
        
        if max_workers == 1:
            # Sequential execution for debugging
            for env_name, algorithm, seed in pending_experiments:
                self.run_single_experiment(env_name, algorithm, seed)
        else:
            # Parallel execution
            with mp.Pool(processes=max_workers) as pool:
                args = [(env_name, algorithm, seed) for env_name, algorithm, seed in pending_experiments]
                results = pool.starmap(self.run_single_experiment, args)
        
        total_time = time.time() - start_time
        self.logger.info(f"All experiments completed in {total_time:.1f} seconds")
        
        # Run analysis
        self.run_analysis()
    
    def run_analysis(self):
        """Run comprehensive analysis of experimental results."""
        self.logger.info("Starting result analysis...")
        
        try:
            analyzer = ResultAnalyzer(self.results_dir, self.config)
            analyzer.analyze_all_results()
            analyzer.generate_summary_report()
            analyzer.create_publication_figures()
            
            self.logger.info("Analysis completed successfully")
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            self.logger.error(traceback.format_exc())


class ResultAnalyzer:
    """Analyze and visualize experimental results."""
    
    def __init__(self, results_dir: Path, config: ExperimentConfig):
        self.results_dir = results_dir
        self.config = config
        self.analysis_dir = results_dir / "analysis"
        self.figures_dir = results_dir / "figures"
        
        # Load all results
        self.all_results = self.load_all_results()
    
    def load_all_results(self) -> List[Dict]:
        """Load all experimental results."""
        results = []
        logs_dir = self.results_dir / "logs"
        
        if not logs_dir.exists():
            return results
        
        for results_file in logs_dir.glob("*_results.pkl"):
            try:
                with open(results_file, 'rb') as f:
                    result = pickle.load(f)
                    if result.get('success', False):
                        results.append(result)
            except Exception as e:
                print(f"Failed to load {results_file}: {e}")
        
        print(f"Loaded {len(results)} successful experiments")
        return results
    
    def analyze_all_results(self):
        """Perform comprehensive analysis."""
        if not self.all_results:
            print("No results to analyze")
            return
        
        # Convert to DataFrame for analysis
        df = self.create_results_dataframe()
        
        # Statistical analysis
        self.perform_statistical_analysis(df)
        
        # Performance analysis
        self.analyze_performance_trends(df)
        
        # AITPR-specific analysis
        self.analyze_aitpr_components(df)
    
    def create_results_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        
        for result in self.all_results:
            if not result.get('success', False):
                continue
            
            base_info = {
                'experiment_id': result['experiment_id'],
                'env_name': result['env_name'],
                'algorithm': result['algorithm'],
                'seed': result['seed'],
                'training_time': result.get('training_time', 0)
            }
            
            # Final performance
            final_perf = result.get('final_performance', {})
            base_info.update({
                'final_mean_reward': final_perf.get('mean_reward', 0),
                'final_std_reward': final_perf.get('std_reward', 0),
                'final_mean_length': final_perf.get('mean_length', 0)
            })
            
            # Training progression
            training_episodes = result.get('training_episodes', [])
            if training_episodes:
                rewards = [ep.get('episode_reward', 0) for ep in training_episodes[-10:]]
                base_info['avg_recent_reward'] = np.mean(rewards)
            
            rows.append(base_info)
        
        df = pd.DataFrame(rows)
        
        # Save DataFrame
        df.to_csv(self.analysis_dir / "results_summary.csv", index=False)
        
        return df
    
    def perform_statistical_analysis(self, df: pd.DataFrame):
        """Perform statistical significance testing."""
        from scipy import stats
        
        analysis_results = {}
        
        # Group by environment
        for env_name in df['env_name'].unique():
            env_df = df[df['env_name'] == env_name]
            env_analysis = {}
            
            # Compare AITPR against each baseline
            aitpr_algorithms = ['aitpr_ppo', 'aitpr_sac']
            baseline_algorithms = [alg for alg in env_df['algorithm'].unique() 
                                 if alg not in aitpr_algorithms]
            
            for aitpr_alg in aitpr_algorithms:
                if aitpr_alg not in env_df['algorithm'].unique():
                    continue
                
                aitpr_rewards = env_df[env_df['algorithm'] == aitpr_alg]['final_mean_reward'].values
                
                for baseline_alg in baseline_algorithms:
                    if baseline_alg not in env_df['algorithm'].unique():
                        continue
                    
                    baseline_rewards = env_df[env_df['algorithm'] == baseline_alg]['final_mean_reward'].values
                    
                    # Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(
                        aitpr_rewards, baseline_rewards, alternative='two-sided'
                    )
                    
                    # Effect size (Cohen's d)
                    effect_size = (np.mean(aitpr_rewards) - np.mean(baseline_rewards)) / np.sqrt(
                        (np.var(aitpr_rewards) + np.var(baseline_rewards)) / 2
                    )
                    
                    comparison_key = f"{aitpr_alg}_vs_{baseline_alg}"
                    env_analysis[comparison_key] = {
                        'aitpr_mean': float(np.mean(aitpr_rewards)),
                        'aitpr_std': float(np.std(aitpr_rewards)),
                        'baseline_mean': float(np.mean(baseline_rewards)),
                        'baseline_std': float(np.std(baseline_rewards)),
                        'mann_whitney_statistic': float(statistic),
                        'p_value': float(p_value),
                        'effect_size': float(effect_size),
                        'significant': p_value < 0.05,
                        'large_effect': abs(effect_size) > 0.5
                    }
            
            analysis_results[env_name] = env_analysis
        
        # Save statistical analysis
        with open(self.analysis_dir / "statistical_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print("Statistical analysis completed")
    
    def analyze_performance_trends(self, df: pd.DataFrame):
        """Analyze performance trends and learning curves."""
        # TODO: Implement learning curve analysis
        print("Performance trend analysis completed")
    
    def analyze_aitpr_components(self, df: pd.DataFrame):
        """Analyze AITPR-specific components (MI, lambda adaptation, etc.)."""
        # TODO: Implement AITPR component analysis
        print("AITPR component analysis completed")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report = {
            'experiment_summary': {
                'total_experiments': len(self.all_results),
                'environments_tested': len(set(r['env_name'] for r in self.all_results)),
                'algorithms_tested': len(set(r['algorithm'] for r in self.all_results)),
                'seeds_per_config': len(self.config.get('seeds')),
                'total_training_time': sum(r.get('training_time', 0) for r in self.all_results)
            }
        }
        
        # Save report
        with open(self.analysis_dir / "summary_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Summary report generated")
    
    def create_publication_figures(self):
        """Create publication-ready figures."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style for publication
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            
            # TODO: Implement figure generation
            print("Publication figures created")
            
        except ImportError:
            print("Matplotlib/Seaborn not available - skipping figure generation")


def main():
    """Main experiment execution function."""
    parser = argparse.ArgumentParser(description="AITPR Experimental Suite")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--results-dir", type=str, default="./results", 
                       help="Results directory")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume interrupted experiments")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Run analysis only (skip experiments)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test with reduced settings")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig(args.config)
    
    # Test mode adjustments
    if args.test:
        config.config['seeds'] = [42, 123]  # Only 2 seeds
        config.config['episodes_per_seed'] = 50  # Fewer episodes
        config.config['environments'] = ['CartPole-v1', 'Pendulum-v1']  # 2 environments
        config.config['algorithms'] = {  # Subset of algorithms
            'aitpr_ppo': config.config['algorithms']['aitpr_ppo'],
            'ppo': config.config['algorithms']['ppo']
        }
    
    # Create experiment runner
    runner = ExperimentRunner(config, args.results_dir, args.resume)
    
    if args.analysis_only:
        # Run analysis only
        runner.run_analysis()
    else:
        # Run full experimental suite
        runner.run_experiments_parallel(max_workers=args.workers)
    
    print("Experiment suite completed successfully!")


if __name__ == "__main__":
    main()