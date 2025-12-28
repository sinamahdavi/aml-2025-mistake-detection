"""
Extract features for CaptainCook4D dataset using new backbones
Usage: python extract_features.py --backbone egovlp --split train
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from core.models.feature_extractors import get_feature_extractor
from dataloader.video_utils import VideoLoader
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory with videos')
    parser.add_argument('--annotations', type=str, required=True,
                       help='Path to annotations JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for features')
    parser.add_argument('--backbone', type=str, default='egovlp',
                       choices=['egovlp', 'perception_encoder'])
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def extract_step_features(video_loader, extractor, video_path, 
                         start_time, end_time):
    """Extract features for a single step"""
    # Load video segment
    frames = video_loader.load_video(video_path, start_time, end_time)
    
    # Segment into clips
    clips = video_loader.segment_into_clips(
        frames, 
        extractor.clip_length,
        stride=extractor.clip_length
    )
    
    # Extract features
    all_features = []
    for clip in clips:
        clip_batch = clip.unsqueeze(0)
        features = extractor.extract_features(clip_batch)
        all_features.append(features)
    
    return torch.cat(all_features, dim=0)

def main():
    args = parse_args()
    
    # Initialize
    video_loader = VideoLoader()
    extractor = get_feature_extractor(args.backbone, device=args.device)
    
    # Load annotations
    with open(args.annotations, 'r') as f:
        annotations = json.load(f)
    
    # Filter by split
    split_data = [a for a in annotations if a['split'] == args.split]
    
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(split_data)} recordings for {args.split}...")
    
    for recording in tqdm(split_data):
        recording_id = recording['recording_id']
        video_path = Path(args.data_root) / recording['video_path']
        
        output_path = output_dir / f"{recording_id}_{args.backbone}.pkl"
        
        if output_path.exists():
            continue
        
        try:
            recording_features = {
                'recording_id': recording_id,
                'backbone': args.backbone,
                'steps': []
            }
            
            for step in recording['steps']:
                step_features = extract_step_features(
                    video_loader,
                    extractor,
                    str(video_path),
                    step['start_time'],
                    step['end_time']
                )
                
                recording_features['steps'].append({
                    'step_id': step['step_id'],
                    'features': step_features,
                    'label': step['label'],
                    'start_time': step['start_time'],
                    'end_time': step['end_time']
                })
            
            with open(output_path, 'wb') as f:
                pickle.dump(recording_features, f)
                
        except Exception as e:
            print(f"Error processing {recording_id}: {e}")
            continue

if __name__ == '__main__':
    main()