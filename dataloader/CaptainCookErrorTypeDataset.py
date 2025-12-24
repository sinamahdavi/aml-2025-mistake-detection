"""
Dataset for Error Type Analysis (Part 2a)
This dataset tracks error types for each step to enable per-error-type evaluation.
"""
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from constants import Constants as const


class CaptainCookErrorTypeDataset(Dataset):
    """
    Dataset that returns step features along with error type information
    for per-error-type performance analysis.
    """

    def __init__(self, config, phase, split):
        self._config = config
        self._backbone = self._config.backbone
        self._phase = phase
        self._split = split

        self._modality = config.modality

        with open('annotations/annotation_json/step_annotations.json', 'r') as f:
            self._annotations = json.load(f)

        with open('annotations/annotation_json/error_annotations.json', 'r') as f:
            self._error_annotations = json.load(f)

        print("Loaded annotations for error type analysis...")

        assert self._phase in ["train", "val", "test"], f"Invalid phase: {self._phase}"

        # Error type mappings
        self._error_type_names = [
            const.TECHNIQUE_ERROR,
            const.PREPARATION_ERROR,
            const.TEMPERATURE_ERROR,
            const.MEASUREMENT_ERROR,
            const.TIMING_ERROR
        ]
        
        self._error_category_name_label_map = {
            const.TECHNIQUE_ERROR: 0,
            const.PREPARATION_ERROR: 1,
            const.TEMPERATURE_ERROR: 2,
            const.MEASUREMENT_ERROR: 3,
            const.TIMING_ERROR: 4
        }

        self._category_name_map = {
            'TechniqueError': const.TECHNIQUE_ERROR,
            'PreparationError': const.PREPARATION_ERROR,
            'TemperatureError': const.TEMPERATURE_ERROR,
            'MeasurementError': const.MEASUREMENT_ERROR,
            'TimingError': const.TIMING_ERROR
        }

        self._build_error_category_labels()

        if self._split == const.STEP_SPLIT:
            self._init_step_split(config, phase)
        else:
            self._init_other_split_from_file(config, phase)

    def _build_error_category_labels(self):
        """Build mapping from recording/step to error categories."""
        self._recording_step_error_labels = {}
        for recording_step_dictionary in self._error_annotations:
            recording_id = recording_step_dictionary['recording_id']
            self._recording_step_error_labels[recording_id] = {}
            for step_annotation_dict in recording_step_dictionary['step_annotations']:
                step_id = step_annotation_dict['step_id']
                # Store set of error type indices (0-4) for this step
                self._recording_step_error_labels[recording_id][step_id] = set()
                if "errors" in step_annotation_dict:
                    for error_dict in step_annotation_dict['errors']:
                        error_tag = error_dict['tag']
                        if error_tag in self._category_name_map:
                            error_name = self._category_name_map[error_tag]
                            error_idx = self._error_category_name_label_map[error_name]
                            self._recording_step_error_labels[recording_id][step_id].add(error_idx)

    def _prepare_recording_step_dictionary(self, recording_id):
        recording_step_dictionary = {}
        for step in self._annotations[recording_id]['steps']:
            step_start_time = step['start_time']
            step_end_time = step['end_time']
            step_id = step['step_id']
            if step_start_time < 0 or step_end_time < 0:
                continue
            
            error_types = self._recording_step_error_labels.get(recording_id, {}).get(step_id, set())

            if recording_step_dictionary.get(step_id) is None:
                recording_step_dictionary[step_id] = []

            recording_step_dictionary[step_id].append(
                (math.floor(step_start_time), math.ceil(step_end_time), step['has_errors'], error_types))
        return recording_step_dictionary

    def _init_step_split(self, config, phase):
        self._recording_ids_file = "recordings_combined_splits.json"
        annotations_file_path = f"./er_annotations/{self._recording_ids_file}"
        with open(f'{annotations_file_path}', 'r') as file:
            self._recording_ids_json = json.load(file)

        self._recording_ids = (self._recording_ids_json['train'] + 
                              self._recording_ids_json['val'] + 
                              self._recording_ids_json['test'])

        self._step_dict = {}
        step_index_id = 0
        for recording_id in self._recording_ids:
            self._normal_step_dict = {}
            self._error_step_dict = {}
            normal_index_id = 0
            error_index_id = 0
            recording_step_dictionary = self._prepare_recording_step_dictionary(recording_id)

            for step_id in recording_step_dictionary.keys():
                if recording_step_dictionary[step_id][0][2]:
                    self._error_step_dict[f'E{error_index_id}'] = (recording_id, recording_step_dictionary[step_id])
                    error_index_id += 1
                else:
                    self._normal_step_dict[f'N{normal_index_id}'] = (recording_id, recording_step_dictionary[step_id])
                    normal_index_id += 1

            np.random.seed(config.seed)
            np.random.shuffle(list(self._normal_step_dict.keys()))
            np.random.shuffle(list(self._error_step_dict.keys()))

            normal_step_indices = list(self._normal_step_dict.keys())
            error_step_indices = list(self._error_step_dict.keys())

            self._split_proportion = [0.75, 0.16, 0.9]

            num_normal_steps = len(normal_step_indices)
            num_error_steps = len(error_step_indices)

            self._split_proportion_normal = [
                int(num_normal_steps * self._split_proportion[0]),
                int(num_normal_steps * (self._split_proportion[0] + self._split_proportion[1]))
            ]
            self._split_proportion_error = [
                int(num_error_steps * self._split_proportion[0]),
                int(num_error_steps * (self._split_proportion[0] + self._split_proportion[1]))
            ]

            if phase == 'train':
                train_indices = (normal_step_indices[:self._split_proportion_normal[0]] + 
                               error_step_indices[:self._split_proportion_error[0]])
                for index_id in train_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id, self._error_step_dict.get(index_id))
                    step_index_id += 1
            elif phase == 'test':
                val_indices = (normal_step_indices[self._split_proportion_normal[0]:self._split_proportion_normal[1]] +
                              error_step_indices[self._split_proportion_error[0]:self._split_proportion_error[1]])
                for index_id in val_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id, self._error_step_dict.get(index_id))
                    step_index_id += 1
            elif phase == 'val':
                test_indices = (normal_step_indices[self._split_proportion_normal[1]:] +
                               error_step_indices[self._split_proportion_error[1]:])
                for index_id in test_indices:
                    self._step_dict[step_index_id] = self._normal_step_dict.get(index_id, self._error_step_dict.get(index_id))
                    step_index_id += 1

    def _init_other_split_from_file(self, config, phase):
        self._recording_ids_file = f"{self._split}_combined_splits.json"
        annotations_file_path = f"./er_annotations/{self._recording_ids_file}"
        with open(f'{annotations_file_path}', 'r') as file:
            self._recording_ids_json = json.load(file)

        self._recording_ids = self._recording_ids_json[phase]
        self._step_dict = {}
        index_id = 0
        for recording_id in self._recording_ids:
            recording_step_dictionary = self._prepare_recording_step_dictionary(recording_id)
            for step_id in recording_step_dictionary.keys():
                self._step_dict[index_id] = (recording_id, recording_step_dictionary[step_id])
                index_id += 1

    def __len__(self):
        assert len(self._step_dict) > 0, "No data found in the dataset"
        return len(self._step_dict)

    def _get_video_features(self, recording_id, step_start_end_list):
        features_path = os.path.join(
            self._config.segment_features_directory, "video", self._backbone,
            f'{recording_id}_360p.mp4_1s_1s.npz')
        features_data = np.load(features_path)
        recording_features = features_data['arr_0']

        step_features = []
        step_has_errors = None
        step_error_types = set()
        
        for step_start_time, step_end_time, has_errors, error_types in step_start_end_list:
            sub_step_features = recording_features[step_start_time:step_end_time, :]
            step_features.append(sub_step_features)
            step_has_errors = has_errors
            step_error_types = step_error_types.union(error_types)
        
        step_features = np.concatenate(step_features, axis=0)
        step_features = torch.from_numpy(step_features).float()
        
        N, d = step_features.shape
        if step_has_errors:
            step_labels = torch.ones(N, 1)
        else:
            step_labels = torch.zeros(N, 1)
        
        # Create error type vector (5 binary values for 5 error types)
        error_type_vector = torch.zeros(5)
        for error_idx in step_error_types:
            error_type_vector[error_idx] = 1.0
        
        features_data.close()
        return step_features, step_labels, error_type_vector

    def __getitem__(self, idx):
        recording_id = self._step_dict[idx][0]
        step_start_end_list = self._step_dict[idx][1]

        step_features, step_labels, error_types = self._get_video_features(recording_id, step_start_end_list)

        return step_features, step_labels, error_types

    def get_error_type_names(self):
        """Return list of error type names."""
        return self._error_type_names


def collate_fn_with_error_types(batch):
    """Collate function that handles error types."""
    step_features, step_labels, error_types = zip(*batch)
    
    step_features = torch.cat(step_features, dim=0)
    step_labels = torch.cat(step_labels, dim=0)
    error_types = torch.stack(error_types, dim=0)  # (batch_size, 5)
    
    return step_features, step_labels, error_types

