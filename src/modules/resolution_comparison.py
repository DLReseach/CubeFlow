import pandas as pd
import numpy as np

from src.modules.utils import get_project_root
from src.modules.utils import get_time
from src.modules.calculate_and_plot import calculate_and_plot


class ResolutionComparison():
    def __init__(self, wandb, config, experiment_name, files_and_dirs):
        super().__init__()
        self.wandb = wandb
        self.config = config
        self.experiment_name = experiment_name
        self.files_and_dirs = files_and_dirs

        self.column_names = [
            'file_number',
            'energy',
            'event_length'
        ]
        self.column_names += ['opponent_' + name + '_error' for name in self.config.comparison_metrics]
        self.column_names += ['own_' + name + '_error' for name in self.config.comparison_metrics]
        self.data = {name: [] for name in self.column_names}

    def match_comparison_and_values(self, df):
        matched_metrics = {}
        for comparison in self.config.comparison_metrics:
            if comparison == 'azimuth' or comparison == 'zenith':
                needed_targets = [
                    'true_primary_direction_x',
                    'true_primary_direction_y',
                    'true_primary_direction_z'
                ]
                needed_own = [
                    'own_primary_direction_x',
                    'own_primary_direction_y',
                    'own_primary_direction_z'
                ]
                converted_predictions = self.convert_to_spherical(
                    df[needed_own].values
                )[comparison]
                converted_truth = self.convert_to_spherical(
                    df[needed_targets].values
                )[comparison]
                normalized_comparisons = self.convert_to_signed_angle(
                    df['opponent_' + comparison].values,
                    comparison
                )
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = converted_predictions
                matched_metrics[comparison]['truth'] = converted_truth
                matched_metrics[comparison]['opponent'] = normalized_comparisons
            elif comparison == 'energy':
                needed_targets = [
                    'true_primary_energy'
                ]
                needed_own = [
                    'own_primary_energy'
                ]
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = np.power(10, df[needed_own].values).flatten()
                matched_metrics[comparison]['truth'] = np.power(10, df[needed_targets].values).flatten()
                matched_metrics[comparison]['opponent'] = df['opponent_' + comparison].values.flatten()
            elif comparison == 'time':
                needed_targets = [
                    'true_primary_time'
                ]
                needed_own = [
                    'own_primary_time'
                ]
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = df[needed_own].values.flatten()
                matched_metrics[comparison]['truth'] = df[needed_targets].values.flatten()
                matched_metrics[comparison]['opponent'] = df['opponent_' + comparison].values.flatten()
        return matched_metrics

    def convert_to_spherical(self, values):
        x = values[:, 0]
        y = values[:, 1]
        z = values[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return {'azimuth': phi, 'zenith': theta}

    def convert_to_signed_angle(self, angles, angle_type):
        if angle_type == 'azimuth':
            signed_angles = [
                entry if entry < np.pi else entry - 2 * np.pi for entry in angles
            ]
            reversed_angles = (
                [entry - np.pi if entry > 0 else entry + np.pi for entry in signed_angles]
            )
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
            delta = np.where(
                abs(difference) > np.pi,
                - 2 * np.sign(difference) * np.pi + difference,
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

    def calculate_errors(self, matched_metrics):
        opponent_error = {}
        own_error = {}
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
            self.data['opponent_' + metric + '_error'] = opponent_error[metric]
            self.data['own_' + metric + '_error'] = own_error[metric]
        
        

    def testing_ended(self, train_true_energy=None, train_event_length=None):
        print('{}: Loading predictions file'.format(get_time()))
        file_name = self.files_and_dirs['run_root'].joinpath('comparison_dataframe_parquet.gzip')
        predictions_df = pd.read_parquet(file_name, engine='fastparquet')
        self.data['file_number'] = predictions_df.file_number.values
        self.data['energy'] = predictions_df.energy.values
        self.data['event_length'] = predictions_df.event_length.values
        print('{}: Matching metrics'.format(get_time()))
        matched_metrics = self.match_comparison_and_values(predictions_df)
        print('{}: Calculating errors'.format(get_time()))
        self.calculate_errors(matched_metrics)
        error_df = pd.DataFrame().from_dict(self.data)
        print('{}: Saving errors file'.format(get_time()))
        file_name = self.files_and_dirs['run_root'].joinpath('error_dataframe_parquet.gzip')
        error_df.to_parquet(
            str(file_name),
            compression='gzip'
        )
        print('{}: Starting calculate_and_plot'.format(get_time()))
        calculate_and_plot(
            self.files_and_dirs,
            self.config,
            self.wandb,
            dom_plots=False,
            use_train_dists=False,
            only_use_metrics=None,
            legends=True,
            reso_hists=False
        )
