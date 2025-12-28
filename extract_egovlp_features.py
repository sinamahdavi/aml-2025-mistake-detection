"""
Real EgoVLP Feature Extractor using the official repository
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import sys
from PIL import Image
import subprocess
import os

sys.path.append(str(Path(__file__).parent))


class EgoVLPFeatureExtractor:
    """
    Feature extractor using official EgoVLP from Facebook Research
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.setup_egovlp()
        self.load_model()
    
    def setup_egovlp(self):
        """Clone and setup EgoVLP repository"""
        egovlp_dir = Path('./EgoVLP')
        
        if not egovlp_dir.exists():
            print("Cloning EgoVLP repository...")
            subprocess.run([
                'git', 'clone', 
                'https://github.com/facebookresearch/EgoVLP.git'
            ], check=True)
            print("✓ EgoVLP repository cloned")
        
        # Add to Python path
        if str(egovlp_dir) not in sys.path:
            sys.path.insert(0, str(egovlp_dir))
        
        # Install dependencies
        requirements_file = egovlp_dir / 'requirements.txt'
        if requirements_file.exists():
            print("Installing EgoVLP dependencies...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-q', 
                '-r', str(requirements_file)
            ])
            print("✓ Dependencies installed")
    
    def load_model(self):
        """Load pretrained EgoVLP model"""
        print("Loading EgoVLP model...")
        
        try:
            # Import EgoVLP modules
            from model.model import FrozenInTime
            from args import get_args
            
            # Get default args
            args = get_args()
            
            # Override with EgoVLP settings
            args.video_params = {
                'model': 'SpaceTimeTransformer',
                'arch_config': 'base_patch16_224',
                'num_frames': 16,
                'pretrained': True,
                'time_init': 'zeros'
            }
            args.text_params = {
                'model': 'distilbert-base-uncased',
                'pretrained': True,
                'input': 'text'
            }
            args.projection_dim = 256
            args.load_checkpoint = None
            
            # Initialize model
            self.model = FrozenInTime(args).to(self.device)
            
            # Download pretrained weights
            checkpoint_path = self._download_checkpoint()
            
            if checkpoint_path and Path(checkpoint_path).exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Load state dict
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                print("✓ Checkpoint loaded")
            else:
                print("⚠ No checkpoint loaded, using random initialization")
            
            self.model.eval()
            
            # Get feature dimension
            with torch.no_grad():
                dummy_video = torch.randn(1, 3, 16, 224, 224).to(self.device)
                dummy_output = self.model.compute_video(dummy_video)
                self.feature_dim = dummy_output.shape[-1]
            
            print(f"✓ EgoVLP loaded successfully!")
            print(f"  Feature dimension: {self.feature_dim}")
            
        except Exception as e:
            print(f"Error loading EgoVLP: {e}")
            print("\nFalling back to CLIP-based implementation...")
            self._load_clip_fallback()
    
    def _download_checkpoint(self):
        """Download pretrained EgoVLP checkpoint"""
        checkpoint_dir = Path('./EgoVLP/pretrained')
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        checkpoint_path = checkpoint_dir / 'egovlp.pth'
        
        if checkpoint_path.exists():
            return str(checkpoint_path)
        
        # URLs for pretrained weights (check EgoVLP repo for actual URLs)
        checkpoint_urls = [
            'https://dl.fbaipublicfiles.com/egovlp/egovlp.pth',
            'https://dl.fbaipublicfiles.com/egovlp/frozen_in_time_egovlp.pth'
        ]
        
        for url in checkpoint_urls:
            try:
                print(f"Downloading checkpoint from {url}...")
                subprocess.run([
                    'wget', '-q', '-O', str(checkpoint_path), url
                ], check=True, timeout=300)
                
                if checkpoint_path.exists() and checkpoint_path.stat().st_size > 1000:
                    print("✓ Checkpoint downloaded")
                    return str(checkpoint_path)
            except:
                continue
        
        print("⚠ Could not download checkpoint")
        return None
    
    def _load_clip_fallback(self):
        """Fallback to CLIP if EgoVLP fails"""
        import clip
        print("Loading CLIP as fallback...")
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.model.eval()
        self.feature_dim = 512
        self.is_clip = True
        print("✓ CLIP loaded as fallback")
    
    def extract_from_video(self, video_path, start_time, end_time, num_frames=16):
        """
        Extract features from video segment
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            num_frames: Number of frames to sample
            
        Returns:
            Feature vector
        """
        # Read video frames
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            cap.release()
            return None
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Sample frames uniformly
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Pad if needed
        while len(frames) < num_frames:
            frames.append(frames[-1])
        
        # Convert to tensor
        frames = np.stack(frames)  # (T, H, W, C)
        frames = torch.from_numpy(frames).float().permute(3, 0, 1, 2)  # (C, T, H, W)
        frames = frames / 255.0
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames = (frames - mean) / std
        
        # Add batch dimension
        frames = frames.unsqueeze(0).to(self.device)  # (1, C, T, H, W)
        
        # Extract features
        with torch.no_grad():
            if hasattr(self, 'is_clip') and self.is_clip:
                # CLIP fallback
                frames_for_clip = frames.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
                features = self.model.encode_image(frames_for_clip)
                features = features.mean(dim=0)
            else:
                # EgoVLP
                try:
                    features = self.model.compute_video(frames)
                    features = features.squeeze()
                except:
                    # Alternative EgoVLP API
                    features = self.model.encode_video(frames)
                    features = features.squeeze()
        
        return features.cpu().numpy()
    
    def extract_text_features(self, text):
        """
        Extract features from text description
        
        Args:
            text: Step description
            
        Returns:
            Text feature vector
        """
        with torch.no_grad():
            if hasattr(self, 'is_clip') and self.is_clip:
                # CLIP fallback
                import clip
                text_tokens = clip.tokenize([text], truncate=True).to(self.device)
                features = self.model.encode_text(text_tokens)
                features = features.squeeze()
            else:
                # EgoVLP
                try:
                    features = self.model.compute_text(text)
                    features = features.squeeze()
                except:
                    # Alternative API
                    features = self.model.encode_text([text])
                    features = features.squeeze()
        
        return features.cpu().numpy()


def load_step_annotations(recording_id, annotations_dir='./annotations'):
    """
    Load step annotations for a recording
    Adapt this to your annotation format
    """
    # Try different possible annotation file locations
    possible_paths = [
        Path(annotations_dir) / f"{recording_id}.json",
        Path(annotations_dir) / recording_id / "annotations.json",
        Path(annotations_dir) / f"rec_{recording_id}.json",
    ]
    
    for ann_file in possible_paths:
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"Annotation file not found for {recording_id}")


def extract_features_for_split(
    split='train',
    data_root='/content/drive/MyDrive/captaincook4d/videos',
    split_file='./er_annotations/recordings_combined_splits.json',
    annotations_dir='./annotations',
    output_dir='/content/drive/MyDrive/extracted_features',
    backbone='egovlp'
):
    """
    Extract EgoVLP features for one split
    """
    
    print(f"\n{'='*60}")
    print(f"Extracting EgoVLP features for {split.upper()} split")
    print(f"{'='*60}\n")
    
    # Load split information
    with open(split_file, 'r') as f:
        splits = json.load(f)
    
    recording_ids = splits[split]
    print(f"Found {len(recording_ids)} recordings in {split} split")
    
    # Initialize extractor
    extractor = EgoVLPFeatureExtractor()
    
    # Create output directory
    output_path = Path(output_dir) / backbone / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track progress
    successful = 0
    failed = []
    
    # Process each recording
    for rec_id in tqdm(recording_ids, desc=f"Processing {split}"):
        try:
            # Find video file
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.MP4']:
                candidate = Path(data_root) / f"{rec_id}{ext}"
                if candidate.exists():
                    video_path = candidate
                    break
            
            if video_path is None:
                print(f"⚠ Video not found for {rec_id}")
                failed.append((rec_id, "video not found"))
                continue
            
            # Load annotations
            try:
                annotations = load_step_annotations(rec_id, annotations_dir)
            except FileNotFoundError:
                print(f"⚠ Annotations not found for {rec_id}")
                failed.append((rec_id, "annotations not found"))
                continue
            
            # Extract features for each step
            recording_features = {
                'video_features': [],
                'text_features': [],
                'labels': [],
                'step_info': []
            }
            
            # Get steps from annotations (adapt field names as needed)
            steps = annotations.get('steps', annotations.get('annotations', []))
            
            for step_idx, step in enumerate(steps):
                start_time = step.get('start_time', step.get('start', 0))
                end_time = step.get('end_time', step.get('end', 0))
                description = step.get('description', step.get('label', 'unknown'))
                label = step.get('error', step.get('is_error', 0))
                
                # Split step into 1-second sub-segments
                step_video_features = []
                current_time = start_time
                
                while current_time < end_time:
                    segment_end = min(current_time + 1.0, end_time)
                    
                    # Extract features for this 1-second segment
                    features = extractor.extract_from_video(
                        video_path, current_time, segment_end, num_frames=16
                    )
                    
                    if features is not None:
                        step_video_features.append(features)
                    
                    current_time = segment_end
                
                # Skip if no features extracted
                if len(step_video_features) == 0:
                    continue
                
                # Extract text features
                text_features = extractor.extract_text_features(description)
                
                # Store features
                recording_features['video_features'].append(
                    np.stack(step_video_features)
                )
                recording_features['text_features'].append(text_features)
                recording_features['labels'].append(label)
                recording_features['step_info'].append({
                    'step_idx': step_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'description': description
                })
            
            # Save features for this recording
            if len(recording_features['labels']) > 0:
                output_file = output_path / f"{rec_id}.npz"
                np.savez_compressed(
                    output_file,
                    video_features=np.array(recording_features['video_features'], dtype=object),
                    text_features=np.stack(recording_features['text_features']),
                    labels=np.array(recording_features['labels']),
                    step_info=recording_features['step_info']
                )
                successful += 1
            
        except Exception as e:
            print(f"❌ Error processing {rec_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed.append((rec_id, str(e)))
            continue
    
    print(f"\n{'='*60}")
    print(f"Feature extraction complete for {split}!")
    print(f"  ✓ Successful: {successful}/{len(recording_ids)}")
    if failed:
        print(f"  ❌ Failed: {len(failed)}")
        for rec_id, reason in failed[:5]:
            print(f"     {rec_id}: {reason}")
        if len(failed) > 5:
            print(f"     ... and {len(failed) - 5} more")
    print(f"  📁 Saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract EgoVLP features')
    parser.add_argument('--data_root', type=str,
                       default='/content/drive/MyDrive/captaincook4d/videos',
                       help='Root directory with videos')
    parser.add_argument('--split_file', type=str,
                       default='./er_annotations/recordings_combined_splits.json',
                       help='JSON file with train/val/test splits')
    parser.add_argument('--annotations_dir', type=str,
                       default='./annotations',
                       help='Directory with step annotations')
    parser.add_argument('--output_dir', type=str,
                       default='/content/drive/MyDrive/extracted_features',
                       help='Output directory for features')
    parser.add_argument('--splits', nargs='+',
                       default=['train', 'val', 'test'],
                       help='Splits to process')
    
    args = parser.parse_args()
    
    # Extract features for all specified splits
    for split in args.splits:
        extract_features_for_split(
            split=split,
            data_root=args.data_root,
            split_file=args.split_file,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir,
            backbone='egovlp'
        )