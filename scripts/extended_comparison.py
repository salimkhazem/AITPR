#!/usr/bin/env python3
"""
Extended baseline comparison for AITPR across diverse RL environments.

This script runs AITPR vs baselines on a comprehensive suite of environments
to provide strong empirical evidence for NeurIPS 2026 submission.

Usage:
    python scripts/extended_comparison.py --phase 1  # Core environments
    python scripts/extended_comparison.py --phase 2  # Advanced environments  
    python scripts/extended_comparison.py --phase all  # Complete suite
"""

import argparse
import time
from pathlib import Path
import gymnasium as gym
from scripts.baseline_comparison import BaselineComparison


class ExtendedComparison(BaselineComparison):
    """Extended comparison across more diverse environments."""
    
    def __init__(self, results_dir: Path = Path("results/extended_comparison")):
        super().__init__(results_dir)
        
        # Phase 1: Core proven environments (quick validation)
        self.core_environments = [
            'CartPole-v1',              # Proven: AITPR +15%
            'MountainCarContinuous-v0', # Proven: AITPR +220%  
            'Acrobot-v1',               # Validated
            'Pendulum-v1',              # Validated
        ]
        
        # Phase 2: Advanced challenging environments  
        self.advanced_environments = [
            'MountainCar-v0',          # Sparse reward discrete
            'LunarLander-v3',          # Complex dynamics, sparse reward
            'InvertedPendulum-v4',     # Balancing, unstable dynamics
            'BipedalWalker-v3',        # Locomotion, high-dimensional
        ]
        
        # Phase 3: Specialized environments (if time permits)
        self.specialized_environments = [
            'CartPoleSwingUp-v1',      # Harder CartPole variant
            'HalfCheetah-v5',          # High-performance continuous
            'Hopper-v5',               # Complex locomotion
        ]
    
    def run_phase_1(self, episodes: int = 50, seeds: int = 3) -> dict:
        """Run core environments - proven AITPR advantages."""
        self.logger.info("=" * 80)
        self.logger.info("PHASE 1: CORE ENVIRONMENTS (Proven AITPR advantages)")
        self.logger.info("=" * 80)
        
        return self.run_comparison(
            environments=self.core_environments,
            episodes=episodes,
            seeds=seeds
        )
    
    def run_phase_2(self, episodes: int = 40, seeds: int = 3) -> dict:
        """Run advanced environments - explore AITPR on harder tasks.""" 
        self.logger.info("=" * 80)
        self.logger.info("PHASE 2: ADVANCED ENVIRONMENTS (Challenging tasks)")
        self.logger.info("=" * 80)
        
        # Test individual environments first to check compatibility
        working_envs = []
        for env_name in self.advanced_environments:
            try:
                env = gym.make(env_name)
                env.close()
                working_envs.append(env_name)
                self.logger.info(f"✓ {env_name} available")
            except Exception as e:
                self.logger.warning(f"✗ {env_name} not available: {e}")
        
        if not working_envs:
            self.logger.error("No advanced environments available!")
            return {}
            
        return self.run_comparison(
            environments=working_envs,
            episodes=episodes, 
            seeds=seeds
        )
    
    def run_phase_3(self, episodes: int = 30, seeds: int = 2) -> dict:
        """Run specialized environments if available."""
        self.logger.info("=" * 80) 
        self.logger.info("PHASE 3: SPECIALIZED ENVIRONMENTS (Extended validation)")
        self.logger.info("=" * 80)
        
        working_envs = []
        for env_name in self.specialized_environments:
            try:
                env = gym.make(env_name)
                env.close()
                working_envs.append(env_name)
                self.logger.info(f"✓ {env_name} available")
            except Exception as e:
                self.logger.warning(f"✗ {env_name} not available: {e}")
        
        if not working_envs:
            self.logger.warning("No specialized environments available, skipping Phase 3")
            return {}
            
        return self.run_comparison(
            environments=working_envs,
            episodes=episodes,
            seeds=seeds
        )
    
    def run_all_phases(self, episodes_per_phase: list = [50, 40, 30], seeds: int = 3) -> dict:
        """Run complete extended comparison across all phases."""
        
        self.logger.info("🚀 STARTING EXTENDED AITPR COMPARISON")
        self.logger.info(f"📊 Total environments: {len(self.core_environments + self.advanced_environments + self.specialized_environments)}")
        self.logger.info(f"⚡ Algorithms: {len(self.algorithms)}")
        self.logger.info(f"🎯 Seeds per experiment: {seeds}")
        
        start_time = time.time()
        all_results = {}
        
        # Phase 1: Core (most important for paper)
        phase1_results = self.run_phase_1(episodes_per_phase[0], seeds)
        all_results['phase_1'] = phase1_results
        
        # Phase 2: Advanced (expand evidence)  
        phase2_results = self.run_phase_2(episodes_per_phase[1], seeds)
        all_results['phase_2'] = phase2_results
        
        # Phase 3: Specialized (if time permits)
        if len(episodes_per_phase) > 2:
            phase3_results = self.run_phase_3(episodes_per_phase[2], seeds)
            all_results['phase_3'] = phase3_results
        
        total_time = time.time() - start_time
        
        # Generate comprehensive summary
        summary = self._generate_extended_summary(all_results, total_time)
        
        # Save extended results
        self._save_extended_results(all_results, summary)
        
        return {
            'all_results': all_results,
            'summary': summary,
            'total_time': total_time
        }
    
    def _generate_extended_summary(self, all_results: dict, total_time: float) -> dict:
        """Generate comprehensive summary across all phases."""
        
        summary = {
            'total_runtime': total_time,
            'phases_completed': list(all_results.keys()),
            'total_environments': 0,
            'total_experiments': 0,
            'aitpr_wins': 0,
            'aitpr_improvements': {},
            'best_environments': [],
            'challenging_environments': [],
            'algorithm_rankings': {}
        }
        
        all_env_results = {}
        
        # Combine results from all phases
        for phase_name, phase_results in all_results.items():
            if 'detailed_results' in phase_results:
                phase_detailed = phase_results['detailed_results']
                all_env_results.update(phase_detailed)
        
        summary['total_environments'] = len(all_env_results)
        
        # Analyze AITPR performance across environments
        for env_name, env_results in all_env_results.items():
            
            # Find best AITPR and best baseline
            aitpr_perfs = []
            baseline_perfs = []
            
            for algo_name, result in env_results.items():
                perf = result.get('eval_mean_overall', float('-inf'))
                
                if 'AITPR' in algo_name:
                    aitpr_perfs.append(perf)
                else:
                    baseline_perfs.append(perf)
            
            if aitpr_perfs and baseline_perfs:
                best_aitpr = max(aitpr_perfs) 
                best_baseline = max(baseline_perfs)
                
                # Calculate improvement percentage
                if abs(best_baseline) > 1e-6:
                    improvement = ((best_aitpr - best_baseline) / abs(best_baseline)) * 100
                else:
                    improvement = 0.0
                    
                summary['aitpr_improvements'][env_name] = improvement
                
                # Classify environments
                if improvement > 10:
                    summary['aitpr_wins'] += 1
                    summary['best_environments'].append({
                        'env': env_name,
                        'improvement': improvement,
                        'aitpr_score': best_aitpr,
                        'baseline_score': best_baseline
                    })
                elif improvement < -5:
                    summary['challenging_environments'].append({
                        'env': env_name, 
                        'improvement': improvement,
                        'aitpr_score': best_aitpr,
                        'baseline_score': best_baseline
                    })
        
        # Sort best environments by improvement
        summary['best_environments'].sort(key=lambda x: x['improvement'], reverse=True)
        
        # Count total experiments
        for env_results in all_env_results.values():
            summary['total_experiments'] += len(env_results)
        
        return summary
    
    def _save_extended_results(self, all_results: dict, summary: dict) -> None:
        """Save extended results with comprehensive analysis."""
        
        import json
        
        # Save detailed phase results
        for phase_name, phase_results in all_results.items():
            phase_dir = self.results_dir / phase_name
            phase_dir.mkdir(parents=True, exist_ok=True)
            
            if phase_results:  # Only save non-empty results
                with open(phase_dir / 'results.json', 'w') as f:
                    json_results = self._convert_for_json(phase_results)
                    json.dump(json_results, f, indent=2)
        
        # Save comprehensive summary  
        with open(self.results_dir / 'extended_summary.json', 'w') as f:
            json_summary = self._convert_for_json(summary)
            json.dump(json_summary, f, indent=2)
        
        # Generate extended plots
        self._generate_extended_plots(all_results)
        
        self.logger.info(f"Extended results saved to {self.results_dir}")
        self.logger.info(f"📈 Summary: {len(summary['aitpr_improvements'])} environments tested")
        self.logger.info(f"🏆 AITPR wins: {summary['aitpr_wins']} environments")
        
        # Print top improvements
        if summary['best_environments']:
            self.logger.info("\n🎯 TOP AITPR IMPROVEMENTS:")
            for i, env_info in enumerate(summary['best_environments'][:5]):
                self.logger.info(f"  {i+1}. {env_info['env']}: +{env_info['improvement']:.1f}%")
    
    def _generate_extended_plots(self, all_results: dict) -> None:
        """Generate extended visualization plots."""
        
        # Combine all environment results for comprehensive plots
        combined_results = {}
        
        for phase_results in all_results.values():
            if 'detailed_results' in phase_results:
                combined_results.update(phase_results['detailed_results'])
        
        if combined_results:
            # Use parent class plotting with combined results
            self._generate_plots(combined_results)
            
            # Generate phase-specific plots
            self._plot_phase_comparison(all_results)
    
    def _plot_phase_comparison(self, all_results: dict) -> None:
        """Plot comparison across different phases."""
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract AITPR improvements by phase
        phase_improvements = {}
        
        for phase_name, phase_results in all_results.items():
            if 'summary' in phase_results:
                improvements = phase_results['summary'].get('performance_comparison', {})
                phase_improvements[phase_name] = []
                
                for env_name, env_perf in improvements.items():
                    if 'improvement_percent' in env_perf:
                        phase_improvements[phase_name].append(env_perf['improvement_percent'])
        
        if phase_improvements:
            # Create phase comparison plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            phases = list(phase_improvements.keys())
            phase_means = [np.mean(impr) if impr else 0 for impr in phase_improvements.values()]
            phase_stds = [np.std(impr) if len(impr) > 1 else 0 for impr in phase_improvements.values()]
            
            bars = ax.bar(phases, phase_means, yerr=phase_stds, capsize=5, 
                         color=['green', 'orange', 'red'][:len(phases)], alpha=0.7)
            
            ax.set_title('AITPR Performance by Environment Phase', fontsize=16, fontweight='bold')
            ax.set_ylabel('Average Improvement (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, mean in zip(bars, phase_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.1f}%', ha='center', va='bottom', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'phase_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Main entry point for extended comparison."""
    parser = argparse.ArgumentParser(description='Extended AITPR baseline comparison')
    parser.add_argument('--phase', choices=['1', '2', '3', 'all'], default='1',
                       help='Which phase to run (1=core, 2=advanced, 3=specialized, all=complete)')
    parser.add_argument('--episodes', type=int, nargs='+', default=[50, 40, 30],
                       help='Episodes per phase [phase1, phase2, phase3]')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of random seeds')
    parser.add_argument('--results-dir', type=str, default='results/extended_comparison',
                       help='Results directory')
    
    args = parser.parse_args()
    
    # Initialize extended comparison
    comparison = ExtendedComparison(Path(args.results_dir))
    
    start_time = time.time()
    
    if args.phase == '1':
        print("🎯 Running Phase 1: Core Environments (Proven AITPR advantages)")
        results = comparison.run_phase_1(args.episodes[0], args.seeds)
        
    elif args.phase == '2':
        print("🚀 Running Phase 2: Advanced Environments")
        results = comparison.run_phase_2(args.episodes[1] if len(args.episodes) > 1 else 40, args.seeds)
        
    elif args.phase == '3':
        print("🏆 Running Phase 3: Specialized Environments")
        results = comparison.run_phase_3(args.episodes[2] if len(args.episodes) > 2 else 30, args.seeds)
        
    elif args.phase == 'all':
        print("🌟 Running Complete Extended Comparison")
        results = comparison.run_all_phases(args.episodes, args.seeds)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("EXTENDED COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Total runtime: {total_time/3600:.2f} hours")
    
    if args.phase == 'all' and 'summary' in results:
        summary = results['summary']
        print(f"📊 Environments tested: {summary['total_environments']}")
        print(f"🏆 AITPR wins: {summary['aitpr_wins']} environments")
        print(f"📈 Average improvement: {np.mean(list(summary['aitpr_improvements'].values())):.1f}%")
        
        if summary['best_environments']:
            print("\n🎯 Top AITPR improvements:")
            for i, env in enumerate(summary['best_environments'][:3]):
                print(f"  {i+1}. {env['env']}: +{env['improvement']:.1f}%")


if __name__ == "__main__":
    import numpy as np
    main()