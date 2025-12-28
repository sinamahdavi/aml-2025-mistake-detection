import wandb
from base import fetch_model_name, train_step_test_step_dataset_base, train_sub_step_test_step_dataset_base, \
    train_model_base
from core.config import Config
from core.utils import init_logger_and_wandb
from constants import Constants as const


def train_sub_step_test_step_er(config):
    train_loader, val_loader, test_loader = train_sub_step_test_step_dataset_base(config)
    train_model_base(train_loader, val_loader, config)


def train_step_test_step_er(config):
    train_loader, val_loader, test_loader = train_step_test_step_dataset_base(config)
    train_model_base(train_loader, val_loader, config, test_loader=test_loader)


def main():
    conf = Config()
    conf.task_name = const.ERROR_RECOGNITION
    
    # ============ NEW: Configure backbone and feature dimension ============
    # Map backbone names to their feature dimensions
    BACKBONE_FEATURE_DIMS = {
        'omnivore': 2048,
        'slowfast': 2304,
        'x3d': 2048,
        '3dresnet': 512,
        'imagebind': 1024,
        'egovlp': 768,              # NEW: EgoVLP features
        'perception_encoder': 1024,  # NEW: PerceptionEncoder features
        'videomae': 768,
    }
    
    # Set feature dimension based on selected backbone
    if hasattr(conf, 'backbone'):
        if conf.backbone in BACKBONE_FEATURE_DIMS:
            conf.feature_dim = BACKBONE_FEATURE_DIMS[conf.backbone]
            print(f"\n{'='*60}")
            print(f"Using backbone: {conf.backbone}")
            print(f"Feature dimension: {conf.feature_dim}")
            print(f"{'='*60}\n")
        else:
            available = ', '.join(BACKBONE_FEATURE_DIMS.keys())
            raise ValueError(
                f"Unknown backbone '{conf.backbone}'.\n"
                f"Available backbones: {available}"
            )
    else:
        # Default fallback for backward compatibility
        print("Warning: No backbone specified, using default feature_dim from config")
        if not hasattr(conf, 'feature_dim'):
            conf.feature_dim = 2048
            print(f"Set default feature_dim: {conf.feature_dim}")
    # ======================================================================
    
    if conf.model_name is None:
        m_name = fetch_model_name(conf)
        conf.model_name = m_name

    if conf.enable_wandb:
        init_logger_and_wandb(conf)

    train_step_test_step_er(conf)

    if conf.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()