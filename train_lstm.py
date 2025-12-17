"""
Training Script for LSTM/GRU Baseline (Part 2b)
Train LSTM or GRU models for error recognition and compare with V1 (MLP) and V2 (Transformer).
"""
import wandb
from base import fetch_model_name, train_step_test_step_dataset_base, train_model_base
from core.config import Config
from core.utils import init_logger_and_wandb
from constants import Constants as const


def train_lstm_er(config):
    """Train LSTM model for error recognition."""
    train_loader, val_loader, test_loader = train_step_test_step_dataset_base(config)
    train_model_base(train_loader, val_loader, config, test_loader=test_loader)


def main():
    conf = Config()
    conf.task_name = const.ERROR_RECOGNITION
    
    # Override variant to LSTM if not specified via command line
    if conf.variant not in [const.LSTM_VARIANT, const.GRU_VARIANT]:
        print(f"Note: Using variant from command line: {conf.variant}")
    
    if conf.model_name is None:
        m_name = fetch_model_name(conf)
        conf.model_name = m_name

    print("="*60)
    print(f"Training {conf.variant} model for Error Recognition")
    print(f"Backbone: {conf.backbone}")
    print(f"Split: {conf.split}")
    print(f"Learning Rate: {conf.lr}")
    print(f"Epochs: {conf.num_epochs}")
    print("="*60)

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    train_lstm_er(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

