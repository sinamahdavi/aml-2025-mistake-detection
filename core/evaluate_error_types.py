"""
Evaluation Script for Error Type Analysis (Part 2a)
Analyzes model performance on different error types:
- Technique Error
- Preparation Error  
- Temperature Error
- Measurement Error
- Timing Error
"""
import argparse
from dataclasses import dataclass
from typing import Optional
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm

from base import fetch_model
from constants import Constants as const
from dataloader.CaptainCookErrorTypeDataset import CaptainCookErrorTypeDataset, collate_fn_with_error_types


@dataclass
class Config(object):
    backbone: str = "omnivore"
    modality: str = "video"
    phase: str = "train"
    segment_length: int = 1
    segment_features_directory: str = "data/"
    ckpt_directory: str = "/data/rohith/captain_cook/checkpoints/"
    split: str = "recordings"
    batch_size: int = 1
    test_batch_size: int = 1
    ckpt: Optional[str] = None
    seed: int = 1000
    device: str = "cuda"
    variant: str = const.TRANSFORMER_VARIANT
    task_name: str = const.ERROR_RECOGNITION


ERROR_TYPE_NAMES = [
    "Technique Error",
    "Preparation Error",
    "Temperature Error", 
    "Measurement Error",
    "Timing Error"
]


def evaluate_per_error_type(model, test_loader, criterion, device, threshold=0.6):
    """
    Evaluate model performance per error type.
    
    Returns:
        Dictionary containing overall metrics and per-error-type metrics
    """
    model.eval()
    
    all_targets = []
    all_outputs = []
    all_error_types = []
    
    test_step_start_end_list = []
    counter = 0
    
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for data, target, error_types in test_loader_tqdm:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            sigmoid_output = output.sigmoid()
            all_outputs.append(sigmoid_output.detach().cpu().numpy().reshape(-1))
            all_targets.append(target.detach().cpu().numpy().reshape(-1))
            all_error_types.append(error_types.numpy())  # (1, 5)
            
            test_step_start_end_list.append((counter, counter + data.shape[0]))
            counter += data.shape[0]
    
    # Flatten outputs and targets
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    all_error_types = np.concatenate(all_error_types, axis=0)  # (num_steps, 5)
    
    # Calculate step-level predictions and targets
    all_step_outputs = []
    all_step_targets = []
    step_error_types = []
    
    for i, (start, end) in enumerate(test_step_start_end_list):
        step_output = all_outputs[start:end]
        step_target = all_targets[start:end]
        
        # Normalize step outputs
        if end - start > 1:
            prob_range = np.max(step_output) - np.min(step_output)
            if prob_range > 0:
                step_output = (step_output - np.min(step_output)) / prob_range
        
        mean_step_output = np.mean(step_output)
        step_target_label = 1 if np.mean(step_target) > 0.95 else 0
        
        all_step_outputs.append(mean_step_output)
        all_step_targets.append(step_target_label)
        step_error_types.append(all_error_types[i])
    
    all_step_outputs = np.array(all_step_outputs)
    all_step_targets = np.array(all_step_targets)
    step_error_types = np.array(step_error_types)  # (num_steps, 5)
    
    # Debug: Check what error types we actually have
    print(f"\nDEBUG: Total steps evaluated: {len(step_error_types)}")
    print(f"DEBUG: Steps with any error type: {np.sum(np.sum(step_error_types, axis=1) > 0)}")
    print(f"DEBUG: Error type counts: {np.sum(step_error_types, axis=0)}")
    print(f"DEBUG: Steps with errors (target=1): {np.sum(all_step_targets == 1)}")
    if np.sum(np.sum(step_error_types, axis=1) > 0) == 0 and np.sum(all_step_targets == 1) > 0:
        print(f"\n⚠️  WARNING: Steps have errors (target=1) but NO error type annotations!")
        print(f"   This means the test set has binary error labels but no error TYPE tags.")
        print(f"   Per-error-type analysis requires error type annotations in error_annotations.json.")
    
    # Global normalization
    prob_range = np.max(all_step_outputs) - np.min(all_step_outputs)
    if prob_range > 0:
        all_step_outputs = (all_step_outputs - np.min(all_step_outputs)) / prob_range
    
    # Calculate overall metrics
    pred_labels = (all_step_outputs > threshold).astype(int)
    
    overall_metrics = {
        'accuracy': accuracy_score(all_step_targets, pred_labels) * 100,
        'precision': precision_score(all_step_targets, pred_labels, zero_division=0) * 100,
        'recall': recall_score(all_step_targets, pred_labels, zero_division=0) * 100,
        'f1': f1_score(all_step_targets, pred_labels, zero_division=0) * 100,
        'auc': roc_auc_score(all_step_targets, all_step_outputs) * 100 if len(np.unique(all_step_targets)) > 1 else 0
    }
    
    # Calculate per-error-type metrics
    per_error_type_metrics = {}
    
    for error_idx, error_name in enumerate(ERROR_TYPE_NAMES):
        # Find steps that have this specific error type
        error_mask = step_error_types[:, error_idx] == 1
        
        if np.sum(error_mask) == 0:
            per_error_type_metrics[error_name] = {
                'count': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'auc': 0
            }
            continue
        
        # Get predictions and targets for this error type
        error_outputs = all_step_outputs[error_mask]
        error_targets = all_step_targets[error_mask]
        error_preds = pred_labels[error_mask]
        
        # Also include some normal (no-error) samples for comparison
        no_error_mask = all_step_targets == 0
        if np.sum(no_error_mask) > 0:
            # Combine error samples with normal samples for AUC calculation
            combined_outputs = np.concatenate([error_outputs, all_step_outputs[no_error_mask]])
            combined_targets = np.concatenate([error_targets, all_step_targets[no_error_mask]])
            combined_preds = np.concatenate([error_preds, pred_labels[no_error_mask]])
        else:
            combined_outputs = error_outputs
            combined_targets = error_targets
            combined_preds = error_preds
        
        # Calculate metrics for this error type
        metrics = {
            'count': int(np.sum(error_mask)),
            'accuracy': accuracy_score(combined_targets, combined_preds) * 100,
            'precision': precision_score(combined_targets, combined_preds, zero_division=0) * 100,
            'recall': recall_score(error_targets, error_preds, zero_division=0) * 100,  # Recall on error samples only
            'f1': f1_score(combined_targets, combined_preds, zero_division=0) * 100,
        }
        
        # Calculate AUC if we have both classes
        if len(np.unique(combined_targets)) > 1:
            metrics['auc'] = roc_auc_score(combined_targets, combined_outputs) * 100
        else:
            metrics['auc'] = 0
        
        per_error_type_metrics[error_name] = metrics
    
    return overall_metrics, per_error_type_metrics


def print_results(overall_metrics, per_error_type_metrics, config):
    """Print formatted results."""
    print("\n" + "="*80)
    print(f"ERROR TYPE ANALYSIS - {config.variant} ({config.backbone}) - {config.split} split")
    print("="*80)
    
    print("\nOVERALL METRICS:")
    print("-"*40)
    print(f"  Accuracy:  {overall_metrics['accuracy']:.2f}%")
    print(f"  Precision: {overall_metrics['precision']:.2f}%")
    print(f"  Recall:    {overall_metrics['recall']:.2f}%")
    print(f"  F1 Score:  {overall_metrics['f1']:.2f}%")
    print(f"  AUC:       {overall_metrics['auc']:.2f}%")
    
    print("\nPER ERROR TYPE METRICS:")
    print("-"*80)
    print(f"{'Error Type':<25} {'Count':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("-"*80)
    
    for error_name, metrics in per_error_type_metrics.items():
        print(f"{error_name:<25} {metrics['count']:>8} {metrics['accuracy']:>7.2f}% {metrics['precision']:>7.2f}% "
              f"{metrics['recall']:>7.2f}% {metrics['f1']:>7.2f}% {metrics['auc']:>7.2f}%")
    
    print("="*80 + "\n")


def save_results_to_csv(overall_metrics, per_error_type_metrics, config, output_dir="results/error_type_analysis"):
    """Save results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{config.variant}_{config.backbone}_{config.split}_error_type_analysis.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        # Write overall metrics
        f.write("Overall Metrics\n")
        f.write("Metric,Value\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric},{value:.2f}\n")
        
        f.write("\nPer Error Type Metrics\n")
        f.write("Error Type,Count,Accuracy,Precision,Recall,F1,AUC\n")
        for error_name, metrics in per_error_type_metrics.items():
            f.write(f"{error_name},{metrics['count']},{metrics['accuracy']:.2f},"
                   f"{metrics['precision']:.2f},{metrics['recall']:.2f},"
                   f"{metrics['f1']:.2f},{metrics['auc']:.2f}\n")
    
    print(f"Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance per error type")
    parser.add_argument("--split", type=str, choices=[const.STEP_SPLIT, const.RECORDINGS_SPLIT], required=True)
    parser.add_argument("--backbone", type=str, choices=[const.SLOWFAST, const.OMNIVORE], required=True)
    parser.add_argument("--variant", type=str, choices=[const.MLP_VARIANT, const.TRANSFORMER_VARIANT, const.LSTM_VARIANT, const.GRU_VARIANT], required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--save_csv", action="store_true", help="Save results to CSV")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Setup config
    config = Config()
    config.split = args.split
    config.backbone = args.backbone
    config.variant = args.variant
    config.ckpt_directory = args.ckpt
    config.device = args.device if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {config.device}")
    
    # Load model
    model = fetch_model(config)
    model.load_state_dict(torch.load(args.ckpt, map_location=config.device))
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")
    
    # Load test dataset with error types
    test_dataset = CaptainCookErrorTypeDataset(config, const.TEST, config.split)
    
    # Diagnostic: Check if test recordings have error annotations
    import json
    with open('annotations/annotation_json/error_annotations.json', 'r') as f:
        error_annos = json.load(f)
    
    split_file = f"./er_annotations/{config.split}_combined_splits.json"
    if os.path.exists(split_file):
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        test_recording_ids = set(split_data.get('test', []))
        error_anno_recording_ids = set([anno['recording_id'] for anno in error_annos])
        
        overlap = test_recording_ids.intersection(error_anno_recording_ids)
        print("\n" + "="*80)
        print("⚠️  ERROR TYPE ANNOTATION DIAGNOSTIC")
        print("="*80)
        print(f"Test recordings in split: {len(test_recording_ids)}")
        print(f"Recordings with error annotations: {len(error_anno_recording_ids)}")
        print(f"Overlap (test recordings WITH error annotations): {len(overlap)}/{len(test_recording_ids)}")
        
        if len(overlap) == 0:
            print(f"\n❌ PROBLEM: No test recordings have error type annotations!")
            print(f"   This is why all error type counts are zero.")
            print(f"   The test set has errors (model detects them), but no error TYPE tags.")
            print(f"   Solution options:")
            print(f"   1. Use validation set for error type analysis (if it has annotations)")
            print(f"   2. Accept that per-error-type metrics aren't available for test set")
            print(f"   3. Check if error_annotations.json needs to be updated with test recordings")
        else:
            # Check if overlapping recordings actually have error type tags
            steps_with_error_types = 0
            for anno in error_annos:
                if anno['recording_id'] in overlap:
                    for step_anno in anno.get('step_annotations', []):
                        if 'errors' in step_anno and len(step_anno['errors']) > 0:
                            steps_with_error_types += 1
            print(f"Steps with error type tags in overlapping recordings: {steps_with_error_types}")
            if steps_with_error_types == 0:
                print(f"   ⚠️  Even though recordings overlap, no steps have error type tags!")
        print("="*80 + "\n")
    
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_with_error_types)
    
    # Evaluate
    criterion = torch.nn.BCEWithLogitsLoss()
    overall_metrics, per_error_type_metrics = evaluate_per_error_type(
        model, test_loader, criterion, config.device, threshold=args.threshold
    )
    
    # Print results
    print_results(overall_metrics, per_error_type_metrics, config)
    
    # Save to CSV if requested
    if args.save_csv:
        save_results_to_csv(overall_metrics, per_error_type_metrics, config)


if __name__ == "__main__":
    main()