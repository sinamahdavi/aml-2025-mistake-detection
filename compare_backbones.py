"""
Compare model variants (MLP, Transformer, LSTM) across different backbones.
"""
import argparse
import os
import glob
import json
from tabulate import tabulate
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from base import fetch_model, test_er_model
from core.config import Config
from constants import Constants as const
from dataloader.CaptainCookStepDataset import CaptainCookStepDataset, collate_fn, step_sequence_collate_fn


def find_best_checkpoint(variant, backbone, split="recordings"):
    """Find the best checkpoint for a given variant and backbone."""
    pattern = f"checkpoints/error_recognition/{variant}/{backbone}/*.pt"
    
    # First try to find *_best.pt
    best_ckpts = glob.glob(pattern.replace('*.pt', '*_best.pt'))
    if best_ckpts:
        return best_ckpts[0]
    
    # Otherwise, get the latest checkpoint
    all_ckpts = sorted(glob.glob(pattern), key=os.path.getmtime)
    return all_ckpts[-1] if all_ckpts else None


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
    config = Config()
    config.backbone = backbone
    config.variant = variant
    config.split = split
    config.device = device
    config.task_name = const.ERROR_RECOGNITION
    
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
    metrics = test_er_model(
        model, test_loader, criterion, device,
        phase="test", step_normalization=True, sub_step_normalization=True, threshold=threshold
    )
    
    return {
        'variant': variant,
        'backbone': backbone,
        **metrics['step_metrics']
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
        backbones = [const.OMNIVORE, const.SLOWFAST]
    
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
    
    # Group by variant for easier comparison
    print("\n" + "=" * 80)
    print("COMPARISON BY MODEL VARIANT")
    print("=" * 80)
    
    grouped = defaultdict(list)
    for result in results:
        grouped[result['variant']].append(result)
    
    for variant in variants:
        variant_results = grouped.get(variant, [])
        if variant_results:
            print(f"\n{variant}:")
            variant_table = []
            for r in sorted(variant_results, key=lambda x: x['backbone']):
                variant_table.append([
                    r['backbone'],
                    f"{r['accuracy']:.2f}",
                    f"{r['precision']:.2f}",
                    f"{r['recall']:.2f}",
                    f"{r['f1']:.2f}",
                    f"{r['auc']:.2f}"
                ])
            print(tabulate(variant_table, headers=["Backbone", "Accuracy", "Precision", "Recall", "F1", "AUC"],
                         tablefmt="grid", floatfmt=".2f"))
    
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
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare model variants across different backbones")
    parser.add_argument("--split", type=str, default="recordings", 
                       choices=[const.RECORDINGS_SPLIT, const.STEP_SPLIT, const.PERSON_SPLIT, const.ENVIRONMENT_SPLIT],
                       help="Data split to use")
    parser.add_argument("--backbones", type=str, nargs="+", 
                       default=[const.OMNIVORE, const.SLOWFAST],
                       choices=[const.OMNIVORE, const.SLOWFAST, const.X3D, const.RESNET3D, const.IMAGEBIND],
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

