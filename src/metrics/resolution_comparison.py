import torch
import pandas as pd
import numpy as np
import pickle
from utils.utils import get_project_root
from plotting.calculate_and_plot import calculate_and_plot


class ResolutionComparison():
    def __init__(self, wandb, config, experiment_name):
        super().__init__()
        self.wandb = wandb
        self.config = config
        self.experiment_name = experiment_name
        self.column_names = [
            'own_error',
            'opponent_error',
            'true_energy',
            'event_length',
            'metric'
        ]
        self.comparison_df = pd.DataFrame(columns=self.column_names)
        if self.config.gpulab:
            self.device = 'cuda:' + self.config.gpulab_gpus
        elif self.config.gpus > 0:
            self.device = 'cuda:' + self.config.gpus
        elif self.config.gpus == 0:
            self.device = 'cpu'

    def match_comparison_and_values(self, predictions, truth, comparisons):
        matched_metrics = {}
        for comparison in comparisons:
            if comparison == 'azimuth' or comparison == 'zenith':
                needed_targets = [
                    'true_primary_direction_x',
                    'true_primary_direction_y',
                    'true_primary_direction_z'
                ]
                needed_targets_test = all(
                    x in self.config.targets for x in needed_targets
                )
                assert needed_targets_test, \
                    'Targets missing for {} comparison'.format(comparison)
                target_indices = []
                for target in needed_targets:
                    target_indices.append(self.config.targets.index(target))
                converted_predictions = self.convert_to_spherical(
                    predictions[:, target_indices]
                )[comparison]
                converted_truth = self.convert_to_spherical(
                    truth[:, target_indices]
                )[comparison]
                normalized_comparisons = self.convert_to_signed_angle(
                    comparisons[comparison],
                    comparison
                )
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = converted_predictions
                matched_metrics[comparison]['truth'] = converted_truth
                matched_metrics[comparison]['opponent'] = normalized_comparisons.to(self.device)
            elif comparison == 'energy':
                needed_targets = [
                    'true_primary_energy'
                ]
                needed_targets_test = all(
                    x in self.config.targets for x in needed_targets
                )
                assert needed_targets_test, \
                    'Targets missing for {} comparison'.format(comparison)
                target_indices = []
                for target in needed_targets:
                    target_indices.append(self.config.targets.index(target))
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = predictions[:, target_indices].flatten()
                matched_metrics[comparison]['truth'] = truth[:, target_indices].flatten()
                matched_metrics[comparison]['opponent'] = torch.log10(
                    comparisons[comparison]
                )
            elif comparison == 'time':
                needed_targets = [
                    'true_primary_time'
                ]
                needed_targets_test = all(
                    x in self.config.targets for x in needed_targets
                )
                assert needed_targets_test, \
                    'Targets missing for {} comparison'.format(comparison)
                target_indices = []
                for target in needed_targets:
                    target_indices.append(self.config.targets.index(target))
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = predictions[:, target_indices].flatten()
                matched_metrics[comparison]['truth'] = truth[:, target_indices].flatten()
                matched_metrics[comparison]['opponent'] = comparisons[comparison].flatten()
        return matched_metrics

    def convert_to_spherical(self, values):
        x = values[:, 0]
        y = values[:, 1]
        z = values[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / r)
        phi = torch.atan2(y, x)
        return {'azimuth': phi, 'zenith': theta}

    def convert_to_signed_angle(self, angles, angle_type):
        if angle_type == 'azimuth':
            signed_angles = [
                entry if entry < np.pi else entry - 2 * np.pi for entry in angles
            ]
            reversed_angles = torch.tensor(
                [entry - np.pi if entry > 0 else entry + np.pi for entry in signed_angles]
            ).float()
        elif angle_type == 'zenith':
            reversed_angles = np.pi - angles
        else:
            print('Unknown angle type')
        return reversed_angles

    def delta_angle(self, prediction, truth, angle_type):
        x = prediction
        y = truth
        difference = x - y
        if angle_type == 'zenith':
            delta = difference
        elif angle_type == 'azimuth':
            delta = torch.where(
                abs(difference) > np.pi,
                - 2 * torch.sign(difference) * np.pi + difference,
                difference
            )
        return delta

    def delta_energy(self, prediction, truth):
        x = prediction
        y = truth
        difference = (x - y) / y
        return difference

    def delta_time(self, prediction, truth):
        x = prediction
        y = truth
        difference = x - y
        return difference

    def update_values(self, predictions, truth, comparisons, energy, event_length):
        opponent_error = {}
        own_error = {}
        matched_metrics = self.match_comparison_and_values(predictions, truth, comparisons)
        for metric in matched_metrics:
            if metric == 'azimuth' or metric == 'zenith':
                opponent_error[metric] = self.delta_angle(
                    matched_metrics[metric]['opponent'],
                    matched_metrics[metric]['truth'],
                    metric
                )
                own_error[metric] = self.delta_angle(
                    matched_metrics[metric]['own'],
                    matched_metrics[metric]['truth'],
                    metric
                )
            elif metric == 'energy':
                opponent_error[metric] = self.delta_energy(
                    matched_metrics[metric]['opponent'],
                    matched_metrics[metric]['truth']
                )
                own_error[metric] = self.delta_energy(
                    matched_metrics[metric]['own'],
                    matched_metrics[metric]['truth']
                )
            elif metric == 'time':
                opponent_error[metric] = self.delta_time(
                    matched_metrics[metric]['opponent'],
                    matched_metrics[metric]['truth']
                )
                own_error[metric] = self.delta_time(
                    matched_metrics[metric]['own'],
                    matched_metrics[metric]['truth']
                )
        for metric in matched_metrics:
            temp_df = pd.DataFrame(
                data={
                    'own_error': own_error[metric].tolist(),
                    'opponent_error': opponent_error[metric].tolist(),
                    'true_energy': energy.tolist(),
                    'event_length': event_length.tolist(),
                    'metric': [metric] * self.config.batch_size
                }
            )
            self.comparison_df = self.comparison_df.append(
                temp_df,
                ignore_index=True
            )

    def testing_ended(self, train_true_energy, train_event_length):
        TRAIN_DATA_DF = pd.DataFrame(
            list(zip(train_true_energy, train_event_length)),
            columns=['train_true_energy', 'train_event_length']
        )
        PROJECT_ROOT = get_project_root()
        RUN_ROOT = PROJECT_ROOT.joinpath('runs')
        RUN_ROOT.mkdir(exist_ok=True)
        RUN_ROOT = RUN_ROOT.joinpath(self.experiment_name)
        RUN_ROOT.mkdir(exist_ok=True)
        self.comparison_df.to_pickle(
            str(RUN_ROOT) + '/comparison_dataframe.gzip'
        )
        TRAIN_DATA_DF.to_pickle(
            str(RUN_ROOT) + '/train_data.gzip'
        )
        if self.config.wandb:
            self.wandb.save(str(RUN_ROOT) + '/comparison_dataframe.gzip')
            self.wandb.save(str(RUN_ROOT) + '/train_data.gzip')
        calculate_and_plot(
            RUN_ROOT,
            self.config,
            self.wandb,
            dom_plots=False
        )
