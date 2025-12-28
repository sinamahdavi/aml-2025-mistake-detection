import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from constants import Constants as const


class CaptainCookStepDataset(Dataset):

    def __init__(self, config, phase, split):
        self._config = config
        self.backbone = config.backbone
        self._phase = phase
        self._split = split

        self._modality = config.modality

        with open('annotations/annotation_json/step_annotations.json', 'r') as f:
            self._annotations = json.load(f)

        with open('annotations/annotation_json/error_annotations.json', 'r') as f:
            self._error_annotations = json.load(f)

        print("Loaded annotations...... ")

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"

        self._build_error_category_label_name_map()
        self._build_error_category_labels()

        if self._split == const.STEP_SPLIT:
            self._init_step_split(config, phase)
        else:
            self._init_other_split_from_file(config, phase)

    def _build_error_category_label_name_map(self):
        self._error_category_name_label_map = {const.TECHNIQUE_ERROR: 6, const.PREPARATION_ERROR: 2,
                                               const.TEMPERATURE_ERROR: 3, const.MEASUREMENT_ERROR: 4,
                                               const.TIMING_ERROR: 5}

        self._error_category_label_name_map = {6: const.TECHNIQUE_ERROR, 2: const.PREPARATION_ERROR,
                                               3: const.TEMPERATURE_ERROR, 4: const.MEASUREMENT_ERROR,
                                               5: const.TIMING_ERROR}

        self._category_name_map = {
            'TechniqueError': const.TECHNIQUE_ERROR,
            'PreparationError': const.PREPARATION_ERROR,
            'TemperatureError': const.TEMPERATURE_ERROR,
            'MeasurementError': const.MEASUREMENT_ERROR,
            'TimingError': const.TIMING_ERROR
        }

    def _build_error_category_labels(self):
        self._recording_step_error_labels = {}
        for recording_step_dictionary in self._error_annotations:
            recording_id = recording_step_dictionary['recording_id']
            self._recording_step_error_labels[recording_id] = {}
            for step_annotation_dict in recording_step_dictionary['step_annotations']:
                step_id = step_annotation_dict['step_id']
                self._recording_step_error_labels[recording_id][step_id] = set()
                if "errors" not in step_annotation_dict:
                    self._recording_step_error_labels[recording_id][step_id].add(0)
                else:
                    for error_dict in step_annotation_dict['errors']:
                        error_tag = error_dict['tag']
                        if error_tag in self._error_category_name_label_map:
                            error_label = self._error_category_name_label_map[error_tag]
                        else:
                            error_label = 0

                        assert error_label is not None, f"Error label not found for error_tag: {error_tag}"
                        self._recording_step_error_labels[recording_id][step_id].add(error_label)

    def _prepare_recording_step_dictionary(self, recording_id):
        recording_step_dictionary = {}
        for step in self._annotations[recording_id]['steps']:
            step_start_time = step['start_time']
            step_end_time = step['end_time']
            step_id = step['step_id']
            if step_start_time < 0 or step_end_time < 0:
                # Ignore missing steps
                continue
            error_category_labels = self._recording_step_error_labels[recording_id][step_id]

            if recording_step_dictionary.get(step_id) is None:
                recording_step_dictionary[step_id] = []

            recording_step_dictionary[step_id].append(
                (math.floor(step_start_time), math.ceil(step_end_time), step['has_errors'], error_category_labels))
        return recording_step_dictionary

    def _init_step_split(self, config, phase):
        self._recording_ids_file = "recordings_combined_splits.json"
        print(f"Loading recording ids from {self._recording_ids_file}")
        annotations_file_path = f"./er_annotations/{self._recording_ids_file}"
        with open(f'{annotations_file_path}', 'r') as file:
            self._recording_ids_json = json.load(file)

        self._recording_ids = self._recording_ids_json['train'] + self._recording_ids_json['val'] + \
                              self._recording_ids_json['test']

        self._step_dict = {}
        step_index_id = 0
        for recording_id in self._recording_ids:
            self._normal_step_dict = {}
            self._error_step_dict = {}
            normal_index_id = 0
            error_index_id = 0
            # 1. Prepare step_id, list(<start, end>) for the recording_id
            recording_step_dictionary = self._prepare_recording_step_dictionary(recording_id)

            # 2. Add step start and end time list to the step_dict
            for step_id in recording_step_dictionary.keys():
                # If the step has errors, add it to the error_step_dict, else add it to the normal_step_dict
                if recording_step_dictionary[step_id][0][2]:
                    self._error_step_dict[f'E{error_index_id}'] = (recording_id, recording_step_dictionary[step_id])
                    error_index_id += 1
                else:
                    self._normal_step_dict[f'N{normal_index_id}'] = (
                        recording_id, recording_step_dictionary[step_id])
                    normal_index_id += 1

            np.random.seed(config.seed)
            np.random.shuffle(list(self._normal_step_dict.keys()))
            np.random.shuffle(list(self._error_step_dict.keys()))

            normal_step_indices = list(self._normal_step_dict.keys())
            error_step_indices = list(self._error_step_dict.keys())

            self._split_proportion = [0.75, 0.16, 0.9]

            num_normal_steps = len(normal_step_indices)
            num_error_steps = len(error_step_indices)

            self._split_proportion_normal = [int(num_normal_steps * self._split_proportion[0]),
                                             int(num_normal_steps * (
                                                     self._split_proportion[0] + self._split_proportion[1]))]
            self._split_proportion_error = [int(num_error_steps * self._split_proportion[0]),
                                            int(num_error_steps * (
                                                    self._split_proportion[0] + self._split_proportion[1]))]

            if phase == 'train':
                self._train_normal = normal_step_indices[:self._split_proportion_normal[0]]
                self._train_error = error_step_indices[:self._split_proportion_error[0]]
                train_indices = self._train_normal + self._train_error
                for index_id in train_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id,
                                                                                self._error_step_dict.get(index_id))
                    step_index_id += 1
            elif phase == 'test':
                self._val_normal = normal_step_indices[
                                   self._split_proportion_normal[0]:self._split_proportion_normal[1]]
                self._val_error = error_step_indices[
                                  self._split_proportion_error[0]:self._split_proportion_error[1]]
                val_indices = self._val_normal + self._val_error
                for index_id in val_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id,
                                                                                self._error_step_dict.get(index_id))
                    step_index_id += 1
            elif phase == 'val':
                self._test_normal = normal_step_indices[self._split_proportion_normal[1]:]
                self._test_error = error_step_indices[self._split_proportion_error[1]:]
                test_indices = self._test_normal + self._test_error
                for index_id in test_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id,
                                                                                self._error_step_dict.get(index_id))
                    step_index_id += 1

    def _init_other_split_from_file(self, config, phase):
        self._recording_ids_file = f"{self._split}_combined_splits.json"
        annotations_file_path = f"./er_annotations/{self._recording_ids_file}"
        print(f"Loading recording ids from {self._recording_ids_file}")
        with open(f'{annotations_file_path}', 'r') as file:
            self._recording_ids_json = json.load(file)

        self._recording_ids = self._recording_ids_json[phase]
        self._step_dict = {}
        index_id = 0
        for recording_id in self._recording_ids:
            # 1. Prepare step_id, list(<start, end>) for the recording_id
            recording_step_dictionary = self._prepare_recording_step_dictionary(recording_id)

            # 2. Add step start and end time list to the step_dict
            for step_id in recording_step_dictionary.keys():
                self._step_dict[index_id] = (recording_id, recording_step_dictionary[step_id])
                index_id += 1

    def __len__(self):
        assert len(self._step_dict) > 0, "No data found in the dataset"
        return len(self._step_dict)

    def _build_task_specific_features_labels(self, step_features, step_has_errors, step_error_category_labels):
        N, d = step_features.shape
        if self._config.task_name == const.ERROR_RECOGNITION:
            if step_has_errors:
                step_labels = torch.ones(N, 1)
            else:
                step_labels = torch.zeros(N, 1)
            return step_features, step_labels
        elif self._config.task_name == const.EARLY_ERROR_RECOGNITION:
            # Input only half of the step features and labels
            step_features = step_features[:N // 2, :]
            if step_has_errors:
                step_labels = torch.ones(N // 2, 1)
            else:
                step_labels = torch.zeros(N // 2, 1)
            return step_features, step_labels
        elif self._config.task_name == const.ERROR_CATEGORY_RECOGNITION:
            error_category_name = self._category_name_map[self._config.error_category]
            task_error_category_label = self._error_category_name_label_map[error_category_name]
            if task_error_category_label in step_error_category_labels:
                step_labels = torch.ones(N, 1)
            else:
                step_labels = torch.zeros(N, 1)
            return step_features, step_labels

    def _build_modality_step_features_labels(self, recording_features, step_start_end_list):
        # Build step features by concatenating the features of the step from the list
        step_features = []
        step_has_errors = None
        step_error_category_labels = None
        for step_start_time, step_end_time, has_errors, error_category_labels in step_start_end_list:
            sub_step_features = recording_features[step_start_time:step_end_time, :]
            step_features.append(sub_step_features)
            step_has_errors = has_errors
            step_error_category_labels = error_category_labels
        step_features = np.concatenate(step_features, axis=0)
        step_features = torch.from_numpy(step_features).float()

        step_features, step_labels = self._build_task_specific_features_labels(
            step_features,
            step_has_errors,
            step_error_category_labels
        )

        return step_features, step_labels

    # ============================================================================
    # UPDATED: Feature loading for different backbones
    # ============================================================================
    
    def _get_feature_path(self, recording_id):
        """
        Construct feature file path based on backbone
        
        For existing backbones (Omnivore, SlowFast):
            data/video/backbone/recording_id.npz
        
        For new backbones (EgoVLP, etc.):
            data/backbone/phase/recording_id_backbone.pkl
        """
        base_dir = self._config.segment_features_directory
        
        # Legacy path for Omnivore and SlowFast
        if self.backbone in [const.OMNIVORE, const.SLOWFAST]:
            features_path = os.path.join(
                base_dir, 
                "video", 
                self.backbone,
                f'{recording_id}_360p.mp4_1s_1s.npz'
            )
            return features_path
        
        # New path structure for new backbones
        else:
            # Structure: data/backbone/phase/recording_id_backbone.pkl
            features_path = os.path.join(
                base_dir,
                self.backbone,
                self._phase,
                f'{recording_id}_{self.backbone}.pkl'
            )
            return features_path
    
    def _load_legacy_features(self, recording_id):
        """Load features for Omnivore and SlowFast (existing .npz format)"""
        features_path = self._get_feature_path(recording_id)
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Feature file not found: {features_path}\n"
                f"Backbone: {self.backbone}"
            )
        
        features_data = np.load(features_path)
        recording_features = features_data['arr_0']
        features_data.close()
        
        return recording_features
    
    def _load_new_backbone_features(self, recording_id):
        """Load features for new backbones (EgoVLP, etc.) from .pkl format"""
        import pickle
        
        features_path = self._get_feature_path(recording_id)
        
        if not os.path.exists(features_path):
            raise FileNotFoundError(
                f"\nFeature file not found: {features_path}\n"
                f"Backbone: {self.backbone}\n"
                f"Phase: {self._phase}\n"
                f"Recording: {recording_id}\n\n"
                f"Have you extracted features for this backbone?\n"
                f"Run: python extract_features.py --backbone {self.backbone} --split {self._phase}"
            )
        
        try:
            with open(features_path, 'rb') as f:
                recording_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading {features_path}: {e}")
        
        # Extract features from the pickle file
        # The pickle contains: {'recording_id': ..., 'backbone': ..., 'steps': [...]}
        # We need to reconstruct the recording_features array indexed by time
        
        # Get all steps and their features
        steps = recording_data.get('steps', [])
        
        if len(steps) == 0:
            raise ValueError(f"No steps found in {features_path}")
        
        # Find the maximum end time to determine array size
        max_time = 0
        for step in steps:
            end_time = int(math.ceil(step.get('end_time', 0)))
            max_time = max(max_time, end_time)
        
        # Get feature dimension from first step
        first_step_features = steps[0]['features']
        if isinstance(first_step_features, torch.Tensor):
            feat_dim = first_step_features.shape[-1]
            first_step_features = first_step_features.numpy()
        else:
            feat_dim = first_step_features.shape[-1]
        
        # Create time-indexed feature array (same format as legacy)
        # This assumes 1-second segments
        recording_features = np.zeros((max_time, feat_dim), dtype=np.float32)
        
        for step in steps:
            start_time = int(math.floor(step.get('start_time', 0)))
            end_time = int(math.ceil(step.get('end_time', 0)))
            step_features = step['features']
            
            # Convert to numpy if tensor
            if isinstance(step_features, torch.Tensor):
                step_features = step_features.numpy()
            
            # step_features shape: (num_clips, feat_dim)
            # We need to fill the time slots [start_time:end_time]
            num_clips = step_features.shape[0]
            duration = end_time - start_time
            
            if num_clips == duration:
                # Direct assignment if dimensions match
                recording_features[start_time:end_time] = step_features
            elif num_clips < duration:
                # Repeat last clip if we have fewer clips than time slots
                recording_features[start_time:start_time+num_clips] = step_features
                if num_clips < duration:
                    recording_features[start_time+num_clips:end_time] = step_features[-1]
            else:
                # Take first 'duration' clips if we have more clips
                recording_features[start_time:end_time] = step_features[:duration]
        
        return recording_features
    
    def _get_video_features(self, recording_id, step_start_end_list):
        """
        Load features based on backbone type
        """
        # Load features using appropriate method
        if self.backbone in [const.OMNIVORE, const.SLOWFAST]:
            recording_features = self._load_legacy_features(recording_id)
        elif self.backbone in [const.EGOVLP, const.PERCEPTION_ENCODER, const.VIDEOMAE]:
            recording_features = self._load_new_backbone_features(recording_id)
        else:
            raise ValueError(
                f"Unsupported backbone: {self.backbone}\n"
                f"Supported: {[const.OMNIVORE, const.SLOWFAST, const.EGOVLP, const.PERCEPTION_ENCODER, const.VIDEOMAE]}"
            )
        
        # Build step features and labels (same for all backbones)
        step_features, step_labels = self._build_modality_step_features_labels(
            recording_features, 
            step_start_end_list
        )
        
        return step_features, step_labels
    
    # ============================================================================
    # END OF UPDATES
    # ============================================================================

    def __getitem__(self, idx):
        recording_id = self._step_dict[idx][0]
        step_start_end_list = self._step_dict[idx][1]

        step_features = None
        step_labels = None
        
        # REMOVED OLD ASSERTION - now supports all backbones
        step_features, step_labels = self._get_video_features(recording_id, step_start_end_list)

        assert step_features is not None, f"Features not found for recording_id: {recording_id}"
        assert step_labels is not None, f"Labels not found for recording_id: {recording_id}"

        return step_features, step_labels


def collate_fn(batch):
    # batch is a list of tuples, and each tuple is (step_features, step_labels)
    step_features, step_labels = zip(*batch)

    # Stack the step_features and step_labels
    step_features = torch.cat(step_features, dim=0)
    step_labels = torch.cat(step_labels, dim=0)

    return step_features, step_labels