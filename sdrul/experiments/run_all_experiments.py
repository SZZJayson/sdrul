"""
Unified experiment runner for CLDR framework.

This script runs multiple experiments with different configurations
and aggregates results for comparison.

Usage:
    python run_all_experiments.py --config experiments/configs/full_framework.yaml
    python run_all_experiments.py --all  # Run all predefined experiments
    python run_all_experiments.py --ablation  # Run all ablation experiments
"""

import os
import sys
import argparse
import logging
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.phase3.train_full import run_experiment, ContinualLearningMetrics


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'experiment_runner_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
    """Convert config dict to argparse namespace with proper type casting."""
    args_dict = {
        'data_dir': str(config['data']['data_dir']),
        'sub_datasets': list(config['data']['sub_datasets']),
        'seq_len': int(config['data']['seq_len']),
        'batch_size': int(config['data']['batch_size']),
        'num_engines': int(config['data']['num_engines']),
        'learning_rate': float(config['model']['learning_rate']),
        'lambda_distill': float(config['loss_weights']['lambda_distillation']),
        'lambda_replay': float(config['loss_weights']['lambda_replay']),
        'lambda_ewc': float(config['loss_weights']['lambda_ewc']),
        'num_prototypes': int(config['components']['num_prototypes']),
        'epochs_per_task': int(config['training']['epochs_per_task']),
        'experiment_name': str(config['experiment']['name']),
        'log_dir': str(config['output']['log_dir']),
        'run_ablations': False,
    }

    # Add replay-specific config if present
    if 'replay' in config:
        for key, value in config['replay'].items():
            if key not in args_dict:
                args_dict[key] = value

    return argparse.Namespace(**args_dict)


def get_method_from_config(config: Dict[str, Any]) -> str:
    """Determine experiment method from configuration."""
    lambda_distill = config['loss_weights']['lambda_distillation']
    lambda_replay = config['loss_weights']['lambda_replay']
    lambda_ewc = config['loss_weights']['lambda_ewc']

    if lambda_distill > 0 and lambda_replay > 0 and lambda_ewc > 0:
        return 'full'
    elif lambda_distill > 0:
        return 'tcsd_only'
    elif lambda_replay > 0:
        return 'replay_only'
    elif lambda_ewc > 0:
        return 'ewc_only'
    else:
        return 'baseline'


def run_single_experiment(
    config_path: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run a single experiment from config file."""
    logger.info(f'\n{"="*60}')
    logger.info(f'Running experiment from: {config_path}')
    logger.info(f'{"="*60}')

    config = load_config(config_path)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")

    args = config_to_args(config)
    method = get_method_from_config(config)
    logger.info(f"Method: {method}")

    # Run the experiment
    cl_metrics, history = run_experiment(args, logger, method=method)

    return {
        'config_file': config_path,
        'experiment_name': config['experiment']['name'],
        'description': config['experiment']['description'],
        'method': method,
        'summary': cl_metrics.get_summary(),
        'history': history,
        'config': config,
    }


def run_multiple_experiments(
    config_paths: List[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run multiple experiments and aggregate results."""
    all_results = {}

    for config_path in config_paths:
        try:
            result = run_single_experiment(config_path, logger)
            experiment_name = result['experiment_name']
            all_results[experiment_name] = result
        except Exception as e:
            logger.error(f'Experiment {config_path} failed: {e}')
            import traceback
            traceback.print_exc()

    return all_results


def compare_results(results: Dict[str, Any], logger: logging.Logger):
    """Compare results across experiments."""
    logger.info('\n' + '='*60)
    logger.info('EXPERIMENT COMPARISON')
    logger.info('='*60)

    # Create comparison table
    logger.info('\n| Experiment | ACC (RMSE) | BWT (Forget) | Description |')
    logger.info('|------------|------------|--------------|-------------|')

    for exp_name, result in results.items():
        summary = result['summary']
        acc = summary['ACC']
        bwt = summary['BWT']
        desc = result.get('description', 'N/A')
        logger.info(f'| {exp_name} | {acc:.4f} | {bwt:.4f} | {desc} |')

    # Calculate improvements if baseline exists
    if 'baseline' in results and 'full_framework' in results:
        base_acc = results['baseline']['summary']['ACC']
        full_acc = results['full_framework']['summary']['ACC']
        base_bwt = results['baseline']['summary']['BWT']
        full_bwt = results['full_framework']['summary']['BWT']

        logger.info('\nImprovements over baseline:')
        logger.info(f'  ACC: {base_acc - full_acc:+.4f} ({"better" if base_acc > full_acc else "worse"})')
        logger.info(f'  BWT: {base_bwt - full_bwt:+.4f} ({"less forgetting" if base_bwt > full_bwt else "more forgetting"})')

    # Rank by ACC
    ranked = sorted(results.items(), key=lambda x: x[1]['summary']['ACC'])
    logger.info('\nRanking by ACC (lower is better):')
    for i, (exp_name, result) in enumerate(ranked, 1):
        logger.info(f'  {i}. {exp_name}: {result["summary"]["ACC"]:.4f}')


def save_aggregated_results(
    results: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
):
    """Save aggregated results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'aggregated_results_{timestamp}.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f'\nAggregated results saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Run CLDR experiments')

    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        help='Paths to multiple YAML config files'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all predefined experiment configs'
    )
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run all ablation experiment configs'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='experiments/phase3/logs',
        help='Directory for logs'
    )

    args = parser.parse_args()

    logger = setup_logging(args.log_dir)
    logger.info('='*60)
    logger.info('CLDR Experiment Runner')
    logger.info('='*60)

    # Determine which configs to run
    config_dir = 'experiments/configs'
    config_paths = []

    if args.config:
        config_paths = [args.config]
    elif args.configs:
        config_paths = args.configs
    elif args.all:
        # Run all configs
        for config_file in os.listdir(config_dir):
            if config_file.endswith('.yaml'):
                config_paths.append(os.path.join(config_dir, config_file))
    elif args.ablation:
        # Run ablation configs
        ablation_configs = [
            'baseline.yaml',
            'replay_only.yaml',
            'tcsd_only.yaml',
            'ewc_only.yaml',
            'full_framework.yaml',
        ]
        for config_file in ablation_configs:
            config_paths.append(os.path.join(config_dir, config_file))
    else:
        logger.error('Please specify --config, --configs, --all, or --ablation')
        return

    logger.info(f'Running {len(config_paths)} experiment(s)')
    for path in config_paths:
        logger.info(f'  - {path}')

    # Run experiments
    results = run_multiple_experiments(config_paths, logger)

    # Compare results
    if len(results) > 1:
        compare_results(results, logger)

    # Save results
    save_aggregated_results(results, args.log_dir, logger)

    logger.info('\nAll experiments completed!')


if __name__ == '__main__':
    main()
