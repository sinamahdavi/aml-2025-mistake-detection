import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json

class MultiBackboneDataset(Dataset):
    """Dataset that works with any backbone features"""
    
    def __init__(self, features_dir, split, backbone='omnivore'):
        """
        Args:
            features_dir: Path to extracted_features/
            split: 'train', 'val', or 'test'
            backbone: 'omnivore', 'slowfast', 'egovlp', etc.
        """
        self.features_dir = Path(features_dir) / backbone / split
        self.backbone = backbone
        self.split = split
        
        # Get all feature files
        self.feature_files = list(self.features_dir.glob("*.npz"))
        
        # Build index: (file_idx, step_idx)
        self.index = []
        for file_idx, feature_file in enumerate(self.feature_files):
            data = np.load(feature_file, allow_pickle=True)
            num_steps = len(data['labels'])
            for step_idx in range(num_steps):
                self.index.append((file_idx, step_idx))
        
        print(f"Loaded {len(self)} samples from {split} split ({backbone})")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        file_idx, step_idx = self.index[idx]
        
        # Load data
        data = np.load(self.feature_files[file_idx], allow_pickle=True)
        
        video_features = data['video_features'][step_idx]  # (num_segments, feature_dim)
        label = data['labels'][step_idx]
        
        # Convert to tensors
        video_features = torch.from_numpy(video_features).float()
        label = torch.tensor(label, dtype=torch.long)
        
        result = {
            'video_features': video_features,
            'label': label,
            'num_segments': video_features.shape[0]
        }
        
        # Add text features if available (for EgoVLP)
        if 'text_features' in data:
            text_features = data['text_features'][step_idx]
            result['text_features'] = torch.from_numpy(text_features).float()
        
        return result


def collate_fn(batch):
    """Collate function for variable-length sequences"""
    max_segments = max(item['num_segments'] for item in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['video_features'].shape[1]
    
    # Initialize tensors
    video_features = torch.zeros(batch_size, max_segments, feature_dim)
    labels = torch.zeros(batch_size, dtype=torch.long)
    segment_masks = torch.zeros(batch_size, max_segments)
    
    for i, item in enumerate(batch):
        n_seg = item['num_segments']
        video_features[i, :n_seg] = item['video_features']
        labels[i] = item['label']
        segment_masks[i, :n_seg] = 1
    
    result = {
        'video_features': video_features,
        'labels': labels,
        'segment_masks': segment_masks
    }
    
    # Add text features if available
    if 'text_features' in batch[0]:
        text_dim = batch[0]['text_features'].shape[0]
        text_features = torch.zeros(batch_size, text_dim)
        for i, item in enumerate(batch):
            text_features[i] = item['text_features']
        result['text_features'] = text_features
    
    return result


def create_dataloaders(features_dir, backbone, batch_size=16, num_workers=2):
    """Create train, val, test dataloaders"""
    from torch.utils.data import DataLoader
    
    datasets = {
        split: MultiBackboneDataset(features_dir, split, backbone)
        for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        for split in ['train', 'val', 'test']
    }
    
    return dataloaders['train'], dataloaders['val'], dataloaders['test']