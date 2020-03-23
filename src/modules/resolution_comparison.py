import pandas as pd
import numpy as np

from src.modules.utils import get_project_root
from src.modules.utils import get_time
from src.modules.calculate_and_plot import calculate_and_plot


class ResolutionComparison():
    def __init__(self, comparison_metrics, files_and_dirs, comparer_config, saver, reporter=None):
        super().__init__()
        self.comparison_metrics = comparison_metrics
        self.comparison_metrics = self.comparison_metrics + ['angle', 'vertex']
        self.files_and_dirs = files_and_dirs
        self.comparer_config = comparer_config
        self.saver = saver
        self.reporter = reporter
        self.reporting_metrics = ['angle', 'vertex', 'time', 'energy']

        self.data = {}

    def match_comparison_and_values(self, df):
        matched_metrics = {}
        for comparison in self.comparison_metrics:
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
            if comparison == 'angle':
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
            elif comparison == 'x':
                needed_targets = [
                    'true_primary_position_x'
                ]
                needed_own = [
                    'own_primary_position_x'
                ]
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = df[needed_own].values.flatten()
                matched_metrics[comparison]['truth'] = df[needed_targets].values.flatten()
                matched_metrics[comparison]['opponent'] = df['opponent_' + comparison].values.flatten()
            elif comparison == 'y':
                needed_targets = [
                    'true_primary_position_y'
                ]
                needed_own = [
                    'own_primary_position_y'
                ]
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = df[needed_own].values.flatten()
                matched_metrics[comparison]['truth'] = df[needed_targets].values.flatten()
                matched_metrics[comparison]['opponent'] = df['opponent_' + comparison].values.flatten()
            elif comparison == 'z':
                needed_targets = [
                    'true_primary_position_z'
                ]
                needed_own = [
                    'own_primary_position_z'
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
        return {'azimuth': phi, 'zenith': theta, 'r': r}

    def convert_to_cartesian(self, angles):
        phi = angles[0]
        theta = angles[1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z]).reshape(3, -1)

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

    def delta_energy_log(self, prediction, truth):
        x = prediction
        y = truth
        difference = np.log10(x / y)
        return difference

    def delta_time(self, prediction, truth):
        x = prediction
        y = truth
        difference = x - y
        return difference

    def delta_position(self, prediction, truth):
        x = prediction
        y = truth
        difference = x - y
        return difference

    def unit_vector(self, vector):
        unit_vecs = []
        for i in range(vector.shape[1]):
            unit_vecs.append(vector[:, i] / np.linalg.norm(vector[:, i]))
        return np.array(unit_vecs)
    
    def delta_angle_vector(self, prediction, truth):
        x = self.convert_to_cartesian(truth)
        y = self.convert_to_cartesian(prediction)
        x_u = self.unit_vector(x)
        y_u = self.unit_vector(y)
        angles = []
        for i in range(x_u.shape[0]):
            angles.append(np.arccos(np.clip(np.dot(x_u[i, :], y_u[i, :]), -1.0, 1.0)))
        # angle = np.arccos(np.einsum('ij,ij->i', x_u, y_u))
        return np.array(angles)

    def vertex_distance(self, prediction, truth):
        x = prediction
        y = truth
        distances = np.linalg.norm((x - y), axis=0)
        return distances

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
            elif metric == 'x' or metric == 'y' or metric == 'z':
                opponent_error[metric] = self.delta_position(
                    matched_metrics[metric]['opponent'],
                    matched_metrics[metric]['truth']
                )
                own_error[metric] = self.delta_position(
                    matched_metrics[metric]['own'],
                    matched_metrics[metric]['truth']
                )
            self.data['opponent_' + metric + '_error'] = opponent_error[metric]
            self.data['own_' + metric + '_error'] = own_error[metric]

    def calculate_errors_new(self, matched_metrics):
        opponent_error = {}
        own_error = {}
        for metric in self.reporting_metrics:
            if metric == 'angle':
                opponent_error[metric] = self.delta_angle_vector(
                    np.array([matched_metrics['azimuth']['opponent'], matched_metrics['zenith']['opponent']]).reshape(2, -1),
                    np.array([matched_metrics['azimuth']['truth'], matched_metrics['zenith']['truth']]).reshape(2, -1)
                )
                own_error[metric] = self.delta_angle_vector(
                    np.array([matched_metrics['azimuth']['own'], matched_metrics['zenith']['own']]).reshape(2, -1),
                    np.array([matched_metrics['azimuth']['truth'], matched_metrics['azimuth']['truth']]).reshape(2, -1)
                )
            elif metric == 'energy':
                opponent_error[metric] = self.delta_energy_log(
                    matched_metrics[metric]['opponent'],
                    matched_metrics[metric]['truth']
                )
                own_error[metric] = self.delta_energy_log(
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
            elif metric == 'vertex':
                opponent_error[metric] = self.vertex_distance(
                    np.array([matched_metrics['x']['opponent'], matched_metrics['y']['opponent'], matched_metrics['z']['opponent']]).reshape(3, -1),
                    np.array([matched_metrics['x']['truth'], matched_metrics['y']['truth'], matched_metrics['z']['truth']]).reshape(3, -1)
                )
                own_error[metric] = self.vertex_distance(
                    np.array([matched_metrics['x']['own'], matched_metrics['y']['own'], matched_metrics['z']['own']]).reshape(3, -1),
                    np.array([matched_metrics['x']['truth'], matched_metrics['y']['truth'], matched_metrics['z']['truth']]).reshape(3, -1)
                )
            self.data['opponent_' + metric + '_error'] = opponent_error[metric]
            self.data['own_' + metric + '_error'] = own_error[metric]
        
        

    def testing_ended(self, train_true_energy=None, train_event_length=None):
        # print('{}: Loading predictions file'.format(get_time()))
        file_name = self.files_and_dirs['run_root'].joinpath('prediction_dataframe_parquet.gzip')
        predictions_df = pd.read_parquet(file_name, engine='fastparquet')
        self.data['file_number'] = predictions_df.file_number.values
        self.data['energy'] = predictions_df.energy.values
        self.data['event_length'] = predictions_df.event_length.values
        # print('{}: Matching metrics'.format(get_time()))
        matched_metrics = self.match_comparison_and_values(predictions_df)
        # print('{}: Calculating errors'.format(get_time()))
        self.calculate_errors_new(matched_metrics)
        error_df = pd.DataFrame().from_dict(self.data)
        # print('{}: Saving errors file'.format(get_time()))
        common_columns = ['file_number']
        common_columns_renamed = ['event']
        own_error_columns = ['own_' + metric + '_error' for metric in self.reporting_metrics]
        opponent_error_columns = ['opponent_' + metric + '_error' for metric in self.reporting_metrics]
        own_error_renamed_columns = [metric.replace('own_', 'predicted_') for metric in own_error_columns]
        own_error_df = error_df[common_columns + own_error_columns + opponent_error_columns]
        own_error_df.columns = common_columns_renamed + own_error_renamed_columns + opponent_error_columns
        file_name = self.files_and_dirs['run_root'].joinpath('own_error_dataframe_parquet.gzip')
        own_error_df.to_parquet(
            str(file_name),
            compression='gzip'
        )
        file_name = self.files_and_dirs['run_root'].joinpath('error_dataframe_parquet.gzip')
        error_df.to_parquet(
            str(file_name),
            compression='gzip'
        )
        self.saver.on_comparison_end(own_error_df)
        # print('{}: Starting calculate_and_plot'.format(get_time()))
        # calculate_and_plot(
        #     self.files_and_dirs,
        #     dom_plots=self.comparer_config['dom_plots'],
        #     use_train_dists=self.comparer_config['use_train_dists'],
        #     only_use_metrics=self.comparer_config['only_use_metrics'],
        #     legends=self.comparer_config['legends'],
        #     reso_hists=self.comparer_config['reso_hists'],
        #     use_own=self.comparer_config['use_own'],
        #     reporter=self.reporter,
        #     wandb=self.comparer_config['wandb']
        # )