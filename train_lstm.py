"""
Training Script for LSTM/GRU Baseline (Part 2b)
Uses CaptainCook dataset with step-level sequences.
Does NOT rely on base.py DataLoader helpers.
"""

import wandb
from torch.utils.data import DataLoader

from base import fetch_model_name, train_model_base
from core.config import Config
from core.utils import init_logger_and_wandb
from constants import Constants as const

from dataloader.CaptainCookStepDataset import (
    CaptainCookStepDataset,
    step_sequence_collate_fn
)


def train_lstm_er(config):
    """Train LSTM/GRU model for error recognition (Step 2b)."""

    # --- DATASETS (CaptainCook) ---
    train_dataset = CaptainCookStepDataset(
        config=config,
        phase="train",
        split=config.split
    )

    val_dataset = CaptainCookStepDataset(
        config=config,
        phase="val",
        split=config.split
    )

    test_dataset = CaptainCookStepDataset(
        config=config,
        phase="test",
        split=config.split
    )

    # --- DATALOADERS (sequence-aware, NEW) ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=step_sequence_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=step_sequence_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=step_sequence_collate_fn
    )

    # --- TRAIN ---
    train_model_base(
        train_loader,
        val_loader,
        config,
        test_loader=test_loader
    )


def main():
    conf = Config()
    conf.task_name = const.ERROR_RECOGNITION

    # Variant must be LSTM or GRU for this script
    if conf.variant not in [const.LSTM_VARIANT, const.GRU_VARIANT]:
        raise ValueError(
            f"Invalid variant for train_lstm.py: {conf.variant}. "
            f"Use {const.LSTM_VARIANT} or {const.GRU_VARIANT}."
        )

    if conf.model_name is None:
        conf.model_name = fetch_model_name(conf)

    print("=" * 60)
    print(f"Training {conf.variant} model for Error Recognition (Step 2b)")
    print(f"Backbone: {conf.backbone}")
    print(f"Split: {conf.split}")
    print(f"Learning Rate: {conf.lr}")
    print(f"Epochs: {conf.num_epochs}")
    print(f"Device: {conf.device}")
    print("=" * 60)

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    train_lstm_er(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()