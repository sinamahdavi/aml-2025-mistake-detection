"""
Baseline Comparison Script (Part 2b)
Compare performance of V1 (MLP), V2 (Transformer), and V3 (LSTM/GRU) baselines.
"""
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm
from tabulate import tabulate

from base import fetch_model
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn


class EvalConfig:
    """Configuration for evaluation."""
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


def evaluate_model(model, test_loader, device, threshold=0.5):
    """Evaluate a model and return metrics."""
    model.eval()
    
    all_targets = []
    all_outputs = []
    test_step_start_end_list = []
    counter = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))
            
            test_step_start_end_list.append((counter, counter + data.shape[0]))
            counter += data.shape[0]
    
    # Flatten
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    # Step-level aggregation
    all_step_outputs = []
    all_step_targets = []
    
    for start, end in test_step_start_end_list:
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]
        
        # Sub-step normalization
        if end - start > 1:
            prob_range = np.max(step_output) - np.min(step_output)
            if prob_range > 0:
                step_output = (step_output - np.min(step_output)) / prob_range
        
        mean_step_output = np.mean(step_output)
        step_target_label = 1 if np.mean(step_target) > 0.95 else 0
        
        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target_label)
    
    all_step_outputs = np.array(all_step_outputs)
    all_step_targets = np.array(all_step_targets)
    
    # Step normalization
    prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
    if prob_range > 0:
        all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range
    
    # Calculate metrics
    pred_labels = (all_step_outputs > threshold).astype(int)
    
    metrics = {
        'Accuracy': accuracy_score(all_step_targets, pred_labels) * 100,
        'Precision': precision_score(all_step_targets, pred_labels, zero_division=0) * 100,
        'Recall': recall_score(all_step_targets, pred_labels, zero_division=0) * 100,
        'F1': f1_score(all_step_targets, pred_labels, zero_division=0) * 100,
        'AUC': roc_auc_score(all_step_targets, all_step_outputs) * 100 if len(np.unique(all_step_targets)) > 1 else 0
    }
    
    return metrics


def compare_baselines(checkpoints, backbone="omnivore", split="recordings", device="cuda"):
    """
    Compare multiple baseline models.
    
    Args:
        checkpoints: Dictionary mapping variant names to checkpoint paths
                    e.g., {"MLP": "path/to/mlp.pt", "Transformer": "path/to/transformer.pt", "LSTM": "path/to/lstm.pt"}
        backbone: Feature backbone (omnivore or slowfast)
        split: Data split (recordings or step)
        device: Device to use
    """
    # Threshold based on split (from original paper)
    threshold = 0.6 if split == "step" else 0.4
    
    results = []
    
    for variant, ckpt_path in checkpoints.items():
        print(f"\n📊 Evaluating {variant}...")
        
        # Create config
        config = EvalConfig(backbone=backbone, variant=variant, split=split, device=device)
        
        # Load model
        model = fetch_model(config)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        # Load test data
        test_dataset = CaptainCookStepDataset(config, const.TEST, split)
        test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device, threshold=threshold)
        
        results.append({
            'Model': variant,
            'Backbone': backbone,
            **metrics
        })
    
    return results


def print_comparison_table(results, split):
    """Print formatted comparison table."""
    print("\n" + "="*90)
    print(f"BASELINE COMPARISON - {split.upper()} SPLIT")
    print("="*90)
    
    headers = ['Model', 'Backbone', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    table_data = []
    
    for r in results:
        table_data.append([
            r['Model'],
            r['Backbone'],
            f"{r['Accuracy']:.2f}",
            f"{r['Precision']:.2f}",
            f"{r['Recall']:.2f}",
            f"{r['F1']:.2f}",
            f"{r['AUC']:.2f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("="*90 + "\n")


def save_comparison_csv(results, split, output_path="results/baseline_comparison.csv"):
    """Save comparison results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Model,Backbone,Split,Accuracy,Precision,Recall,F1,AUC\n")
        for r in results:
            f.write(f"{r['Model']},{r['Backbone']},{split},"
                   f"{r['Accuracy']:.2f},{r['Precision']:.2f},"
                   f"{r['Recall']:.2f},{r['F1']:.2f},{r['AUC']:.2f}\n")
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline models for error recognition")
    parser.add_argument("--split", type=str, choices=["step", "recordings"], required=True)
    parser.add_argument("--backbone", type=str, choices=["omnivore", "slowfast"], default="omnivore")
    parser.add_argument("--mlp_ckpt", type=str, help="Path to MLP checkpoint")
    parser.add_argument("--transformer_ckpt", type=str, help="Path to Transformer checkpoint")
    parser.add_argument("--lstm_ckpt", type=str, help="Path to LSTM checkpoint")
    parser.add_argument("--gru_ckpt", type=str, help="Path to GRU checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_csv", action="store_true", help="Save results to CSV")
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Build checkpoint dictionary
    checkpoints = {}
    if args.mlp_ckpt:
        checkpoints["MLP"] = args.mlp_ckpt
    if args.transformer_ckpt:
        checkpoints["Transformer"] = args.transformer_ckpt
    if args.lstm_ckpt:
        checkpoints["LSTM"] = args.lstm_ckpt
    if args.gru_ckpt:
        checkpoints["GRU"] = args.gru_ckpt
    
    if not checkpoints:
        print("Error: Please provide at least one checkpoint path")
        return
    
    # Compare baselines
    results = compare_baselines(checkpoints, args.backbone, args.split, device)
    
    # Print results
    print_comparison_table(results, args.split)
    
    # Save to CSV if requested
    if args.save_csv:
        save_comparison_csv(results, args.split)


if __name__ == "__main__":
    main()

