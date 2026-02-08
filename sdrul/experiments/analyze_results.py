"""
Results visualization and analysis for CLDR framework experiments.

This script loads experiment results and creates:
1. Comparison tables
2. Learning curves
3. Forgetting analysis
4. Performance metrics visualization

Usage:
    python analyze_results.py --results experiments/phase3/logs/aggregated_results_*.json
    python analyze_results.py --latest  # Analyze latest results
"""

import os
import sys
import argparse
import logging
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'analysis_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def find_latest_results(log_dir: str) -> Optional[str]:
    """Find the latest aggregated results file."""
    pattern = os.path.join(log_dir, 'aggregated_results_*.json')
    files = glob.glob(pattern)

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def print_comparison_table(results: Dict[str, Any], logger: logging.Logger):
    """Print a comparison table of all experiments."""
    logger.info('\n' + '='*80)
    logger.info('EXPERIMENT COMPARISON TABLE')
    logger.info('='*80)

    # Header
    logger.info(f'{"Experiment":<20} {"ACC (RMSE)":<12} {"BWT":<12} {"FWT":<12} {"Description"}')
    logger.info('-' * 80)

    # Rows
    for exp_name, result in sorted(results.items()):
        if 'summary' not in result:
            continue

        summary = result['summary']
        acc = summary.get('ACC', 0)
        bwt = summary.get('BWT', 0)
        fwt = summary.get('FWT', 0)
        desc = result.get('description', '')

        logger.info(f'{exp_name:<20} {acc:<12.4f} {bwt:<12.4f} {fwt:<12.4f} {desc}')

    logger.info('='*80)


def calculate_improvements(results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Calculate improvements over baseline."""
    if 'baseline' not in results:
        return {}

    baseline = results['baseline']['summary']
    improvements = {}

    for exp_name, result in results.items():
        if exp_name == 'baseline':
            continue

        if 'summary' not in result:
            continue

        summary = result['summary']
        improvements[exp_name] = {
            'acc_improvement': baseline['ACC'] - summary['ACC'],
            'bwt_improvement': baseline['BWT'] - summary['BWT'],
            'acc_relative': (baseline['ACC'] - summary['ACC']) / baseline['ACC'] * 100,
        }

    return improvements


def print_improvements(results: Dict[str, Any], logger: logging.Logger):
    """Print improvements over baseline."""
    improvements = calculate_improvements(results)

    if not improvements:
        logger.info('\nNo baseline found for comparison.')
        return

    logger.info('\n' + '='*80)
    logger.info('IMPROVEMENTS OVER BASELINE')
    logger.info('='*80)
    logger.info(f'{"Experiment":<20} {"ACC Δ":<12} {"ACC %":<12} {"BWT Δ":<12} {"Interpretation"}')
    logger.info('-' * 80)

    for exp_name, imp in improvements.items():
        acc_delta = imp['acc_improvement']
        acc_pct = imp['acc_relative']
        bwt_delta = imp['bwt_improvement']

        # Interpretation
        acc_better = "better" if acc_delta > 0 else "worse"
        bwt_better = "less forgetting" if bwt_delta > 0 else "more forgetting"
        interp = f"ACC {acc_better}, {bwt_better}"

        logger.info(f'{exp_name:<20} {acc_delta:<+12.4f} {acc_pct:<+12.2f}% {bwt_delta:<+12.4f} {interp}')

    logger.info('='*80)


def analyze_forgetting(results: Dict[str, Any], logger: logging.Logger):
    """Analyze forgetting patterns across tasks."""
    logger.info('\n' + '='*80)
    logger.info('FORGETTING ANALYSIS')
    logger.info('='*80)

    for exp_name, result in sorted(results.items()):
        if 'history' not in result or 'task_metrics' not in result['history']:
            continue

        task_metrics = result['history']['task_metrics']
        if not task_metrics:
            continue

        # Get number of tasks
        num_tasks = len(task_metrics)

        # Calculate forgetting for each task
        forgetting_scores = {}

        for task_id in range(num_tasks):
            # Performance when task was first trained
            initial_perf = task_metrics.get(str(task_id), {}).get(str(task_id), 0)

            # Performance after all tasks trained
            final_perf = task_metrics.get(str(num_tasks - 1), {}).get(str(task_id), 0)

            if initial_perf > 0:
                forgetting = final_perf - initial_perf  # Positive = forgetting
                forgetting_scores[f'task_{task_id}'] = forgetting

        if forgetting_scores:
            logger.info(f'\n{exp_name}:')
            logger.info(f'  Average forgetting: {np.mean(list(forgetting_scores.values())):.4f}')
            logger.info(f'  Max forgetting: {max(forgetting_scores.values()):.4f}')
            logger.info(f'  Min forgetting: {min(forgetting_scores.values()):.4f}')

    logger.info('='*80)


def plot_learning_curves(
    results: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
):
    """Plot learning curves for all experiments."""
    if not PLOT_AVAILABLE:
        logger.warning('matplotlib not available, skipping plots')
        return

    os.makedirs(output_dir, exist_ok=True)

    for exp_name, result in results.items():
        if 'history' not in result or 'task_losses' not in result['history']:
            continue

        task_losses = result['history']['task_losses']

        if not task_losses:
            continue

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Total loss over epochs
        ax = axes[0]
        for task_id, losses in task_losses.items():
            if 'total_loss' in losses and isinstance(losses['total_loss'], list):
                epochs = range(1, len(losses['total_loss']) + 1)
                ax.plot(epochs, losses['total_loss'], marker='o', label=f'Task {task_id}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{exp_name}: Total Loss per Task')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss components
        ax = axes[1]
        # Aggregate losses across all tasks
        components = ['supervised_loss', 'distillation_loss', 'replay_loss', 'ewc_loss']
        for comp in components:
            all_values = []
            for losses in task_losses.values():
                if comp in losses and isinstance(losses[comp], list):
                    all_values.extend(losses[comp])
            if all_values:
                ax.plot(all_values, alpha=0.7, label=comp.replace('_', ' ').title())

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{exp_name}: Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Task performance (RMSE)
        ax = axes[2]
        if 'task_metrics' in result['history']:
            task_metrics = result['history']['task_metrics']

            for eval_task in range(len(task_metrics)):
                perf_per_task = []
                for train_task in range(len(task_metrics)):
                    metrics = task_metrics.get(str(train_task), {})
                    rmse = metrics.get(str(eval_task), 0)
                    perf_per_task.append(rmse)

                if perf_per_task:
                    ax.plot(range(len(task_metrics)), perf_per_task, marker='s', label=f'Task {eval_task}')

        ax.set_xlabel('Tasks Trained')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{exp_name}: Performance Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{exp_name}_learning_curves.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f'Saved learning curves: {output_path}')


def plot_comparison_bars(
    results: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
):
    """Plot comparison bar charts."""
    if not PLOT_AVAILABLE:
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics
    exp_names = []
    acc_values = []
    bwt_values = []

    for exp_name, result in sorted(results.items()):
        if 'summary' not in result:
            continue

        summary = result['summary']
        exp_names.append(exp_name)
        acc_values.append(summary.get('ACC', 0))
        bwt_values.append(summary.get('BWT', 0))

    if not exp_names:
        return

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(exp_names))
    width = 0.6

    # ACC comparison
    bars1 = ax1.bar(x, acc_values, width, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('ACC (RMSE)')
    ax1.set_title('Average RMSE (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # BWT comparison
    colors = ['green' if v < 0 else 'red' for v in bwt_values]
    bars2 = ax2.bar(x, bwt_values, width, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('BWT (Backward Transfer)')
    ax2.set_title('Forgetting (Negative = Less Forgetting)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_bars.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f'Saved comparison bars: {output_path}')


def generate_latex_table(results: Dict[str, Any], output_dir: str, logger: logging.Logger):
    """Generate LaTeX table for paper."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'results_table_{timestamp}.tex')

    # Build LaTeX table
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{CLDR Framework Performance Comparison}')
    lines.append('\\label{tab:results}')
    lines.append('\\begin{tabular}{lccc}')
    lines.append('\\toprule')
    lines.append('Method & ACC (RMSE)$\\downarrow$ & BWT$\\uparrow$ & Description \\\\')
    lines.append('\\midrule')

    for exp_name, result in sorted(results.items()):
        if 'summary' not in result:
            continue

        summary = result['summary']
        acc = summary.get('ACC', 0)
        bwt = summary.get('BWT', 0)
        desc = result.get('description', '')

        # Format: escape underscores, wrap desc in \text{}
        formatted_name = exp_name.replace('_', '\\_')
        formatted_desc = desc.replace('_', '\\_')

        # Bold best results
        acc_str = f'{acc:.2f}'
        bwt_str = f'{bwt:.2f}'

        lines.append(f'{formatted_name} & {acc_str} & {bwt_str} & {formatted_desc} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f'Saved LaTeX table: {output_path}')


def generate_markdown_report(
    results: Dict[str, Any],
    output_dir: str,
    logger: logging.Logger,
):
    """Generate markdown report."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'report_{timestamp}.md')

    lines = []
    lines.append('# CLDR Framework Experiment Results')
    lines.append(f'\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # Summary table
    lines.append('## Summary')
    lines.append('\n| Method | ACC (RMSE) | BWT | FWT | Description |')
    lines.append('|--------|------------|-----|-----|-------------|')

    for exp_name, result in sorted(results.items()):
        if 'summary' not in result:
            continue

        summary = result['summary']
        lines.append(
            f'| {exp_name} | {summary["ACC"]:.4f} | {summary["BWT"]:.4f} | '
            f'{summary["FWT"]:.4f} | {result.get("description", "")} |'
        )

    # Improvements
    improvements = calculate_improvements(results)
    if improvements:
        lines.append('\n## Improvements over Baseline')
        lines.append('\n| Method | ACC Δ | ACC % | BWT Δ |')
        lines.append('|--------|-------|-------|-------|')

        for exp_name, imp in improvements.items():
            lines.append(
                f'| {exp_name} | {imp["acc_improvement"]:+.4f} | '
                f'{imp["acc_relative"]:+.2f}% | {imp["bwt_improvement"]:+.4f} |'
            )

    lines.append('\n---')
    lines.append('*Lower RMSE and BWT indicate better performance.*')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f'Saved markdown report: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Analyze CLDR experiment results')

    parser.add_argument(
        '--results',
        type=str,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Use latest results file'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='experiments/phase3/logs',
        help='Directory containing results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/phase3/analysis',
        help='Directory for analysis outputs'
    )

    args = parser.parse_args()

    logger = setup_logging(args.log_dir)
    logger.info('='*60)
    logger.info('CLDR Results Analysis')
    logger.info('='*60)

    # Determine results file
    if args.latest:
        results_path = find_latest_results(args.log_dir)
        if results_path is None:
            logger.error('No results files found')
            return
        logger.info(f'Using latest results: {results_path}')
    elif args.results:
        results_path = args.results
    else:
        logger.error('Please specify --results or --latest')
        return

    # Load results
    logger.info(f'Loading results from: {results_path}')
    results = load_results(results_path)

    # Extract experiment results if in aggregated format
    if 'results' not in results:
        # Assume the file contains multiple experiment keys
        experiment_results = results
    else:
        experiment_results = results['results']

    # Print comparison table
    print_comparison_table(experiment_results, logger)

    # Print improvements
    print_improvements(experiment_results, logger)

    # Analyze forgetting
    analyze_forgetting(experiment_results, logger)

    # Create visualizations
    if PLOT_AVAILABLE:
        plot_learning_curves(experiment_results, args.output_dir, logger)
        plot_comparison_bars(experiment_results, args.output_dir, logger)

    # Generate reports
    generate_latex_table(experiment_results, args.output_dir, logger)
    generate_markdown_report(experiment_results, args.output_dir, logger)

    logger.info('\nAnalysis complete!')


if __name__ == '__main__':
    main()
