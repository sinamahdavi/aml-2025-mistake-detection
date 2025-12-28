"""
Main script to extract EgoVLP features from CaptainCook4D videos
Run this ONCE to extract all features, then save to Drive
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

class EgoVLPFeatureExtractor:
    """Simplified feature extractor for Google Colab"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.load_model()
    
    def load_model(self):
        """Load EgoVLP model"""
        try:
            # Try HuggingFace first
            from transformers import AutoModel, AutoProcessor
            print("Loading EgoVLP from HuggingFace...")
            self.model = AutoModel.from_pretrained("Intel/egovlp").to(self.device)
            self.processor = AutoProcessor.from_pretrained("Intel/egovlp")
            self.model.eval()
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please install: pip install transformers")
            raise
    
    def extract_from_video(self, video_path, start_time, end_time):
        """Extract features from video segment"""
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        frames = []
        for frame_idx in range(start_frame, end_frame, max(1, (end_frame - start_frame) // 16)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return None
        
        # Convert to tensor
        frames = np.stack(frames)
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        
        # Extract features
        with torch.no_grad():
            frames_tensor = frames_tensor.to(self.device)
            features = self.model.get_image_features(frames_tensor)
            features = features.mean(dim=0)  # Average over frames
        
        return features.cpu().numpy()
    
    def extract_text_features(self, text):
        """Extract text features"""
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", 
                                   padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)
        return features.cpu().numpy().squeeze()


def extract_features_for_split(split='train', 
                               data_root='/content/drive/MyDrive/captaincook4d/videos',
                               annotations_path='./er_annotations/recordings_combined_splits.json',
                               output_dir='/content/drive/MyDrive/extracted_features'):
    """Extract features for one split"""
    
    print(f"\n{'='*60}")
    print(f"Extracting features for {split.upper()} split")
    print(f"{'='*60}\n")
    
    # Load annotations
    with open(annotations_path, 'r') as f:
        split_data = json.load(f)
    
    recording_ids = split_data[split]
    print(f"Found {len(recording_ids)} recordings in {split} split")
    
    # Initialize extractor
    extractor = EgoVLPFeatureExtractor()
    
    # Create output directory
    output_path = Path(output_dir) / 'egovlp' / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each recording
    for rec_id in tqdm(recording_ids, desc=f"Processing {split}"):
        try:
            # Load recording annotations (you need to adapt this)
            # This is a placeholder - adjust based on your annotation format
            recording_info = load_recording_info(rec_id, annotations_path)
            
            video_path = Path(data_root) / f"{rec_id}.mp4"
            
            if not video_path.exists():
                print(f"Warning: Video not found: {video_path}")
                continue
            
            recording_features = {
                'video_features': [],
                'text_features': [],
                'labels': []
            }
            
            # Process each step
            for step in recording_info['steps']:
                # Extract video features (divide into 1-sec segments)
                step_features = []
                duration = step['end_time'] - step['start_time']
                
                for t in np.arange(step['start_time'], step['end_time'], 1.0):
                    t_end = min(t + 1.0, step['end_time'])
                    features = extractor.extract_from_video(
                        str(video_path), t, t_end
                    )
                    if features is not None:
                        step_features.append(features)
                
                if len(step_features) > 0:
                    recording_features['video_features'].append(
                        np.stack(step_features)
                    )
                    
                    # Extract text features
                    text_features = extractor.extract_text_features(
                        step['description']
                    )
                    recording_features['text_features'].append(text_features)
                    recording_features['labels'].append(step['label'])
            
            # Save features
            output_file = output_path / f"{rec_id}.npz"
            np.savez_compressed(
                output_file,
                video_features=np.array(recording_features['video_features'], dtype=object),
                text_features=np.stack(recording_features['text_features']),
                labels=np.array(recording_features['labels'])
            )
            
        except Exception as e:
            print(f"Error processing {rec_id}: {e}")
            continue
    
    print(f"\n✓ Feature extraction complete for {split}!")
    print(f"  Saved to: {output_path}")


def load_recording_info(recording_id, annotations_path):
    """
    Load recording information from annotations
    YOU NEED TO IMPLEMENT THIS based on your annotation format
    """
    # This is a placeholder - adapt to your actual annotation structure
    # Return format should be:
    # {
    #   'steps': [
    #       {
    #           'start_time': float,
    #           'end_time': float,
    #           'description': str,
    #           'label': int (0 or 1)
    #       },
    #       ...
    #   ]
    # }
    raise NotImplementedError("Implement based on your annotation format")


if __name__ == "__main__":
    # Configuration
    DATA_ROOT = "/content/drive/MyDrive/captaincook4d/videos"
    ANNOTATIONS_PATH = "./er_annotations/recordings_combined_splits.json"
    OUTPUT_DIR = "/content/drive/MyDrive/extracted_features"
    
    # Extract for all splits
    for split in ['train', 'val', 'test']:
        extract_features_for_split(
            split=split,
            data_root=DATA_ROOT,
            annotations_path=ANNOTATIONS_PATH,
            output_dir=OUTPUT_DIR
        )