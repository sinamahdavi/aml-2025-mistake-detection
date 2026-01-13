"""
Compare model variants (MLP, Transformer, LSTM) across different backbones.
"""
import argparse
import os
import glob
import json
from tabulate import tabulate
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader

from base import fetch_model, test_er_model
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn, step_sequence_collate_fn


class EvalConfig:
    """Simple config class for evaluation without argument parsing."""
    def __init__(self, backbone="omnivore", variant="MLP", split="recordings", device="cuda"):
        self.backbone = backbone
        self.modality = "video"
        self.phase = "test"
        self.segment_length = 1
        self.segment_features_directory = "data/"
        self.ckpt_directory = ""
        self.split = split
        self.batch_size = 1
        self.test_batch_size = 1
        self.seed = 1000
        self.device = device
        self.variant = variant
        self.task_name = const.ERROR_RECOGNITION


def find_best_checkpoint(variant, backbone, split="recordings"):
    """Find the best checkpoint for a given variant and backbone."""
    # Try multiple patterns to handle different naming conventions
    patterns = [
        f"checkpoints/error_recognition/{variant}/{backbone}/*_best.pt",
        f"checkpoints/error_recognition/{variant}/{backbone}/*best*.pt",
        f"checkpoints/error_recognition/{variant}/{backbone}/*.pt"
    ]
    
    for pattern in patterns:
        ckpts = glob.glob(pattern)
        if ckpts:
            # If multiple, prefer *_best.pt, then sort by modification time
            best_ckpts = [c for c in ckpts if '_best' in c.lower() or 'best' in c.lower()]
            if best_ckpts:
                return sorted(best_ckpts, key=os.path.getmtime)[-1]
            return sorted(ckpts, key=os.path.getmtime)[-1]
    
    return None


def evaluate_model_with_backbone(variant, backbone, split="recordings", device="cuda", threshold=0.4):
    """Evaluate a model with a specific backbone."""
    # Find checkpoint
    ckpt_path = find_best_checkpoint(variant, backbone, split)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"⚠️  No checkpoint found for {variant} + {backbone}")
        return None
    
    print(f"📊 Evaluating {variant} with {backbone} backbone...")
    print(f"   Checkpoint: {os.path.basename(ckpt_path)}")
    
    # Create config
    config = EvalConfig(backbone=backbone, variant=variant, split=split, device=device)
    
    # Load model
    model = fetch_model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # Load test data
    test_dataset = CaptainCookStepDataset(config, const.TEST, split)
    # Use step_sequence_collate_fn for LSTM/GRU, regular collate_fn for others
    if variant in [const.LSTM_VARIANT, const.GRU_VARIANT]:
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=step_sequence_collate_fn)
    else:
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    
    # Evaluate
    criterion = torch.nn.BCEWithLogitsLoss()
    test_losses, sub_step_metrics, step_metrics = test_er_model(
        model, test_loader, criterion, device,
        phase="test", step_normalization=True, sub_step_normalization=True, threshold=threshold
    )
    
    return {
        'variant': variant,
        'backbone': backbone,
        'accuracy': step_metrics[const.ACCURACY] * 100,
        'precision': step_metrics[const.PRECISION] * 100,
        'recall': step_metrics[const.RECALL] * 100,
        'f1': step_metrics[const.F1] * 100,
        'auc': step_metrics[const.AUC] * 100
    }


def compare_backbones(variants=None, backbones=None, split="recordings", device="cuda", save_csv=True):
    """
    Compare model variants across different backbones.
    
    Args:
        variants: List of model variants to compare (default: ['MLP', 'Transformer', 'LSTM'])
        backbones: List of backbones to compare (default: ['omnivore', 'slowfast'])
        split: Data split to use
        device: Device to use
        save_csv: Whether to save results to CSV
    """
    if variants is None:
        variants = [const.MLP_VARIANT, const.TRANSFORMER_VARIANT, const.LSTM_VARIANT]
    if backbones is None:
        backbones = [const.OMNIVORE, const.SLOWFAST, const.EGOVLP]
    
    threshold = 0.6 if split == "step" else 0.4
    
    print("=" * 80)
    print(f"BACKBONE COMPARISON: {', '.join(variants)} across {', '.join(backbones)}")
    print(f"Split: {split}")
    print("=" * 80)
    
    results = []
    
    for variant in variants:
        for backbone in backbones:
            result = evaluate_model_with_backbone(variant, backbone, split, device, threshold)
            if result:
                results.append(result)
    
    if not results:
        print("\n❌ No results found! Please train models first.")
        return
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("BACKBONE COMPARISON RESULTS")
    print("=" * 80)
    
    # Create table
    table_data = []
    for result in results:
        table_data.append([
            result['variant'],
            result['backbone'],
            f"{result['accuracy']:.2f}",
            f"{result['precision']:.2f}",
            f"{result['recall']:.2f}",
            f"{result['f1']:.2f}",
            f"{result['auc']:.2f}"
        ])
    
    headers = ["Model", "Backbone", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))
    
    # Save to CSV
    if save_csv:
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/backbone_comparison_{split}.csv"
        
        with open(csv_path, 'w') as f:
            # Write header
            f.write("Model,Backbone,Accuracy,Precision,Recall,F1,AUC\n")
            # Write data
            for result in results:
                f.write(f"{result['variant']},{result['backbone']},"
                       f"{result['accuracy']:.4f},{result['precision']:.4f},"
                       f"{result['recall']:.4f},{result['f1']:.4f},{result['auc']:.4f}\n")
        
        print(f"\n✅ Results saved to: {csv_path}")
    
    # Create visualizations
    create_comparison_charts(results, split, save_csv)
    
    return results


def create_comparison_charts(results, split="recordings", save_csv=True):
    """Create visualization charts for backbone comparison."""
    if not results:
        return
    
    os.makedirs("results", exist_ok=True)
    
    # Prepare data for plotting
    variants = sorted(set(r['variant'] for r in results))
    backbones = sorted(set(r['backbone'] for r in results))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data: [variant][backbone] = value
        data = {}
        for variant in variants:
            data[variant] = {}
            for backbone in backbones:
                # Find result for this variant+backbone combination
                result = next((r for r in results if r['variant'] == variant and r['backbone'] == backbone), None)
                data[variant][backbone] = result[metric] if result else 0
        
        # Create grouped bar chart
        x = np.arange(len(variants))
        width = 0.35
        
        if len(backbones) == 2:
            backbone1_values = [data[v][backbones[0]] for v in variants]
            backbone2_values = [data[v][backbones[1]] for v in variants]
            
            bars1 = ax.bar(x - width/2, backbone1_values, width, label=backbones[0].capitalize(), alpha=0.8)
            bars2 = ax.bar(x + width/2, backbone2_values, width, label=backbones[1].capitalize(), alpha=0.8)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom', fontsize=9)
        else:
            # Handle more than 2 backbones
            for i, backbone in enumerate(backbones):
                values = [data[v][backbone] for v in variants]
                offset = width * (i - len(backbones)/2 + 0.5)
                bars = ax.bar(x + offset, values, width, label=backbone.capitalize(), alpha=0.8)
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model Variant', fontsize=11)
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontsize=11)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        max_val = max([r[metric] for r in results] + [50])
        ax.set_ylim([0, max_val * 1.1])
    
    # Remove the last empty subplot
    fig.delaxes(axes[5])
    
    plt.suptitle(f'Backbone Comparison - {split.upper()} Split', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    chart_path = f"results/backbone_comparison_{split}_charts.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Charts saved to: {chart_path}")
    
    # Also create a summary comparison chart
    create_summary_chart(results, split, save_csv)
    
    plt.close()


def create_summary_chart(results, split="recordings", save_csv=True):
    """Create a summary chart showing all metrics for all models."""
    if not results:
        return
    
    variants = sorted(set(r['variant'] for r in results))
    backbones = sorted(set(r['backbone'] for r in results))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data: x-axis will be model+backbone combinations
    x_labels = []
    metric_data = {metric: [] for metric in metrics}
    
    for variant in variants:
        for backbone in backbones:
            result = next((r for r in results if r['variant'] == variant and r['backbone'] == backbone), None)
            if result:
                x_labels.append(f"{variant}\n{backbone}")
                for metric in metrics:
                    metric_data[metric].append(result[metric])
            else:
                x_labels.append(f"{variant}\n{backbone}\n(missing)")
                for metric in metrics:
                    metric_data[metric].append(0)
    
    x = np.arange(len(x_labels))
    width = 0.15
    
    # Create bars for each metric
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i*width, metric_data[metric], width, label=metric.capitalize(), 
                     color=colors[i], alpha=0.8)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Model + Backbone', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'Complete Backbone Comparison - {split.upper()} Split', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    max_val = max([r[m] for r in results for m in metrics] + [50])
    ax.set_ylim([0, max_val * 1.1])
    
    plt.tight_layout()
    
    # Save figure
    chart_path = f"results/backbone_comparison_{split}_summary.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Summary chart saved to: {chart_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare model variants across different backbones")
    parser.add_argument("--split", type=str, default="recordings", 
                       choices=[const.RECORDINGS_SPLIT, const.STEP_SPLIT, const.PERSON_SPLIT, const.ENVIRONMENT_SPLIT],
                       help="Data split to use")
    parser.add_argument("--backbones", type=str, nargs="+", 
                       default=[const.OMNIVORE, const.SLOWFAST, const.EGOVLP],
                       choices=[const.OMNIVORE, const.SLOWFAST, const.X3D, const.RESNET3D, const.IMAGEBIND, const.EGOVLP],
                       help="Backbones to compare")
    parser.add_argument("--variants", type=str, nargs="+",
                       default=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT, const.LSTM_VARIANT],
                       choices=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT, const.LSTM_VARIANT, const.GRU_VARIANT],
                       help="Model variants to compare")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--save_csv", action="store_true", default=True, help="Save results to CSV")
    
    args = parser.parse_args()
    
    compare_backbones(
        variants=args.variants,
        backbones=args.backbones,
        split=args.split,
        device=args.device,
        save_csv=args.save_csv
    )


if __name__ == "__main__":
    main()

