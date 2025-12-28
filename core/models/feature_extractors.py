
#EgoVLP and PerceptionEncoder


import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

class BaseFeatureExtractor(nn.Module):
    """Base class for feature extractors"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.clip_length = None
        self.feature_dim = None
    
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EgoVLPExtractor(BaseFeatureExtractor):
    """EgoVLP feature extractor"""
    
    def __init__(self, device='cuda', model_path=None):
        super().__init__(device)
        self.clip_length = 16
        self.feature_dim = 768
        
        # ImageNet normalization
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        """Load pretrained EgoVLP"""
        try:
            from transformers import AutoModel
            if model_path:
                model = AutoModel.from_pretrained(model_path)
            else:
                model = AutoModel.from_pretrained("qinghonglin/EgoVLP")
            return model.to(self.device)
        except Exception as e:
            print(f"Error loading EgoVLP: {e}")
            raise
    
    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames - self.mean) / self.std
    
    @torch.no_grad()
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) or (T, C, H, W)
        Returns:
            features: (B, D) or (D,)
        """
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        
        frames = frames.to(self.device)
        frames = self.normalize(frames)
        
        # EgoVLP expects (B, C, T, H, W)
        frames = frames.permute(0, 2, 1, 3, 4)
        features = self.model.encode_video(frames)
        
        return features.cpu()


class PerceptionEncoderExtractor(BaseFeatureExtractor):
    """Perception Encoder feature extractor"""
    
    def __init__(self, device='cuda', layer='intermediate', model_path=None):
        super().__init__(device)
        self.clip_length = 8
        self.feature_dim = 1024  # Depends on layer
        self.layer = layer
        
        # CLIP normalization
        self.register_buffer(
            'mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, model_path):
        """Load Perception Encoder"""
        # TODO: Implement based on PE release
        raise NotImplementedError("PE loading to be implemented")
    
    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        return (frames - self.mean) / self.std
    
    @torch.no_grad()
    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.to(self.device)
        frames = self.normalize(frames)
        
        if self.layer == 'intermediate':
            features = self.model.encode_video_intermediate(frames)
        else:
            features = self.model.encode_video(frames)
        
        return features.cpu()


def get_feature_extractor(backbone: str, device='cuda', **kwargs):
    """Factory function to get feature extractor"""
    extractors = {
        'egovlp': EgoVLPExtractor,
        'perception_encoder': PerceptionEncoderExtractor
    }
    
    if backbone not in extractors:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    return extractors[backbone](device=device, **kwargs)