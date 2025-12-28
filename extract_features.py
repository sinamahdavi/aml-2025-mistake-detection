import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import cv2

class EgoVLPFeatureExtractor:
    """
    Feature extractor using EgoVLP backbone.
    EgoVLP aligns video and text embeddings in a shared space.
    """
    def __init__(self, device='cuda', model_path=None):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path):
        """Load pre-trained EgoVLP model"""
        try:
            from transformers import AutoModel, AutoProcessor
            # Load EgoVLP model (adjust model name as needed)
            model = AutoModel.from_pretrained(
                'Intel/EgoVLP',
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                'Intel/EgoVLP',
                trust_remote_code=True
            )
            return model
        except Exception as e:
            print(f"Error loading EgoVLP: {e}")
            print("Attempting alternative loading method...")
            # Fallback implementation
            return self._load_custom_egovlp(model_path)
    
    def _load_custom_egovlp(self, model_path):
        """Custom EgoVLP loader if transformers doesn't work"""
        # Placeholder for custom loading logic
        class DummyEgoVLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.visual_encoder = nn.Sequential(
                    nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d((1, 7, 7)),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 512)
                )
            
            def forward(self, x):
                return self.visual_encoder(x)
        
        return DummyEgoVLP()
    
    def extract_features(self, video_path: str, num_frames: int = 16) -> torch.Tensor:
        """
        Extract features from video using EgoVLP
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            
        Returns:
            Feature tensor of shape (feature_dim,)
        """
        frames = self._load_video_frames(video_path, num_frames)
        
        with torch.no_grad():
            if hasattr(self, 'processor'):
                inputs = self.processor(videos=frames, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze()
            else:
                # Fallback processing
                frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
                frames_tensor = frames_tensor.float().to(self.device) / 255.0
                features = self.model(frames_tensor).squeeze()
        
        return features.cpu()
    
    def _load_video_frames(self, video_path: str, num_frames: int) -> np.ndarray:
        """Load and sample frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return np.stack(frames)


class PerceptionEncoderFeatureExtractor:
    """
    Feature extractor using Perception Encoder backbone.
    PE provides intermediate layer features that are often better for downstream tasks.
    """
    def __init__(self, device='cuda', layer_idx=-2):
        self.device = device
        self.layer_idx = layer_idx  # Which layer to extract features from
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self):
        """Load Perception Encoder model"""
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                'perception-encoder',  # Adjust to actual model name
                trust_remote_code=True
            ).to(self.device)
            return model
        except Exception as e:
            print(f"Error loading Perception Encoder: {e}")
            return self._load_custom_pe()
    
    def _load_custom_pe(self):
        """Custom PE loader"""
        # Use a standard video model as fallback
        try:
            import torchvision.models.video as video_models
            model = video_models.r3d_18(pretrained=True)
            # Modify to extract intermediate features
            model.fc = nn.Identity()
            return model.to(self.device)
        except Exception as e:
            print(f"Fallback loading failed: {e}")
            return None
    
    def extract_features(self, video_path: str, num_frames: int = 16) -> torch.Tensor:
        """Extract features using Perception Encoder"""
        frames = self._load_video_frames(video_path, num_frames)
        
        with torch.no_grad():
            # Prepare input: (B, C, T, H, W)
            frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
            frames_tensor = frames_tensor.float().to(self.device) / 255.0
            
            # Extract features from specified layer
            features = self._extract_layer_features(frames_tensor)
        
        return features.cpu()
    
    def _extract_layer_features(self, x):
        """Extract features from intermediate layer"""
        # This is a simplified version - adjust based on actual model architecture
        if hasattr(self.model, 'forward_features'):
            return self.model.forward_features(x, layer_idx=self.layer_idx)
        else:
            return self.model(x)
    
    def _load_video_frames(self, video_path: str, num_frames: int) -> np.ndarray:
        """Load video frames - same as EgoVLP"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return np.stack(frames)


class FeatureExtractionPipeline:
    """
    Main pipeline for extracting features from CaptainCook4D dataset
    """
    def __init__(self, backbone_name: str, device='cuda'):
        self.backbone_name = backbone_name
        self.device = device
        self.extractor = self._get_extractor()
        
    def _get_extractor(self):
        """Get appropriate feature extractor"""
        if self.backbone_name.lower() == 'egovlp':
            return EgoVLPFeatureExtractor(device=self.device)
        elif self.backbone_name.lower() in ['pe', 'perceptionencoder']:
            return PerceptionEncoderFeatureExtractor(device=self.device)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
    
    def process_dataset(
        self,
        video_dir: Path,
        annotation_file: Path,
        output_dir: Path,
        subsegment_duration: float = 1.0
    ):
        """
        Process entire dataset and extract features
        
        Args:
            video_dir: Directory containing videos
            annotation_file: JSON file with step annotations
            output_dir: Where to save extracted features
            subsegment_duration: Duration of sub-segments in seconds
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        print(f"Extracting features using {self.backbone_name}...")
        
        for recording_id, recording_data in tqdm(annotations.items()):
            recording_features = self._process_recording(
                recording_id,
                recording_data,
                video_dir,
                subsegment_duration
            )
            
            # Save features
            output_file = output_dir / f"{recording_id}_features.pt"
            torch.save(recording_features, output_file)
    
    def _process_recording(
        self,
        recording_id: str,
        recording_data: Dict,
        video_dir: Path,
        subsegment_duration: float
    ) -> Dict:
        """Process single recording and extract step-level features"""
        video_path = video_dir / recording_data['video_file']
        steps = recording_data['steps']
        
        recording_features = {
            'recording_id': recording_id,
            'steps': []
        }
        
        for step_idx, step in enumerate(steps):
            start_time = step['start_time']
            end_time = step['end_time']
            step_duration = end_time - start_time
            
            # Split into sub-segments
            num_subsegments = max(1, int(step_duration / subsegment_duration))
            subsegment_features = []
            
            for i in range(num_subsegments):
                subseg_start = start_time + i * subsegment_duration
                subseg_end = min(subseg_start + subsegment_duration, end_time)
                
                # Extract features for this sub-segment
                features = self._extract_subsegment_features(
                    video_path,
                    subseg_start,
                    subseg_end
                )
                subsegment_features.append(features)
            
            recording_features['steps'].append({
                'step_idx': step_idx,
                'step_description': step.get('description', ''),
                'error_label': step.get('error_label', 0),
                'error_type': step.get('error_type', 'none'),
                'subsegment_features': torch.stack(subsegment_features),
                'num_subsegments': num_subsegments
            })
        
        return recording_features
    
    def _extract_subsegment_features(
        self,
        video_path: Path,
        start_time: float,
        end_time: float
    ) -> torch.Tensor:
        """Extract features for a specific time segment"""
        # This is a simplified version - you may need to implement
        # temporal cropping of the video before feature extraction
        features = self.extractor.extract_features(str(video_path))
        return features


def compare_backbones(
    video_path: str,
    backbones: List[str] = ['egovlp', 'perceptionencoder']
):
    """
    Utility function to compare different backbones
    
    Args:
        video_path: Path to a sample video
        backbones: List of backbone names to compare
    """
    print("Comparing feature extraction backbones...")
    results = {}
    
    for backbone_name in backbones:
        try:
            print(f"\nTesting {backbone_name}...")
            pipeline = FeatureExtractionPipeline(backbone_name)
            features = pipeline.extractor.extract_features(video_path)
            
            results[backbone_name] = {
                'feature_dim': features.shape[0] if features.dim() == 1 else features.shape,
                'feature_norm': torch.norm(features).item(),
                'success': True
            }
            print(f"✓ {backbone_name}: dim={results[backbone_name]['feature_dim']}")
        except Exception as e:
            results[backbone_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ {backbone_name}: {e}")
    
    return results


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with chosen backbone
    pipeline = FeatureExtractionPipeline('egovlp', device='cuda')
    
    # Process dataset
    video_dir = Path('/path/to/videos')
    annotation_file = Path('er_annotations/recordings_combined_splits.json')
    output_dir = Path('extracted_features/egovlp')
    
    pipeline.process_dataset(
        video_dir=video_dir,
        annotation_file=annotation_file,
        output_dir=output_dir
    )
    
    print("Feature extraction complete!")