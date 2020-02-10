import torch
import torch.nn.functional as F
import joblib
import h5py as h5
from pathlib import Path
import pandas as pd
import numpy as np
import math
from PIL import Image
import io
from utils.math_funcs import unit_vector
from utils.utils import get_project_root
from utils.utils import get_time

import matplotlib.pyplot as plt


class ResolutionComparison():
    def __init__(self, wandb, config):
        super().__init__()
        self.wandb = wandb
        self.config = config
        self.column_names = [
            'own_error',
            'opponent_error',
            'true_energy',
            'metric'
        ]
        self.comparison_df = pd.DataFrame(columns=self.column_names)

    def match_comparison_and_values(self, predictions, truth, comparisons):
        matched_metrics = {}
        for comparison in comparisons:
            if comparison == 'azimuth' or comparison == 'zenith':
                needed_targets = [
                    'true_primary_direction_x',
                    'true_primary_direction_y',
                    'true_primary_direction_z'
                ]
                needed_targets_test = all(x in self.config.targets for x in needed_targets)
                assert needed_targets_test, 'Targets missing for {} comparison'.format(comparison)
                target_indices = []
                for target in needed_targets:
                    target_indices.append(self.config.targets.index(target))
                converted_predictions = self.convert_to_spherical(predictions[:, target_indices])[comparison]
                converted_truth = self.convert_to_spherical(truth[:, target_indices])[comparison]
                normalized_comparisons = self.convert_to_signed_angle(comparisons[comparison], comparison)
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = converted_predictions
                matched_metrics[comparison]['truth'] = converted_truth
                matched_metrics[comparison]['opponent'] = normalized_comparisons
            elif comparison == 'energy':
                needed_targets = [
                    'true_primary_energy'
                ]
                needed_targets_test = all(x in self.config.targets for x in needed_targets)
                assert needed_targets_test, 'Targets missing for {} comparison'.format(comparison)
                target_indices = []
                for target in needed_targets:
                    target_indices.append(self.config.targets.index(target))
                log_transformed_comparisons = np.log10(comparisons[comparison])
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = predictions[:, target_indices].flatten()
                matched_metrics[comparison]['truth'] = truth[:, target_indices].flatten()
                matched_metrics[comparison]['opponent'] = log_transformed_comparisons
            elif comparison == 'time':
                needed_targets = [
                    'true_primary_time'
                ]
                needed_targets_test = all(x in self.config.targets for x in needed_targets)
                assert needed_targets_test, 'Targets missing for {} comparison'.format(comparison)
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
        difference = (y - x) / y
        return difference

    def delta_time(self, prediction, truth):
        x = prediction
        y = truth
        difference = x - y
        return difference

    def update_values(self, predictions, truth, comparisons, energy):
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
                    'metric': [metric] * self.config.batch_size
                }
            )
            self.comparison_df = self.comparison_df.append(
                temp_df,
                ignore_index=True
            )

    def calculate_energy_bins(self):
        no_of_bins = 12
        self.comparison_df['binned'] = pd.cut(
            self.comparison_df['true_energy'], no_of_bins
        )
        bins = self.comparison_df.binned.unique()
        return bins

    def convert_iqr_to_sigma(self, quartiles, e_quartiles):
        factor = 1 / 1.349
        sigma = np.abs(quartiles[1] - quartiles[0]) * factor
        e_sigma = factor * np.sqrt(e_quartiles[0]**2 + e_quartiles[1]**2)        
        return sigma, e_sigma

    def estimate_percentile(self, data, percentiles, n_bootstraps=1000):
        data = np.array(data)
        n = data.shape[0]
        data.sort()
        i_means, means = [], []
        i_plussigmas, plussigmas = [], []
        i_minussigmas, minussigmas = [], []
        for percentile in percentiles:
            sigma = np.sqrt(percentile * n * (1 - percentile))
            mean = n * percentile
            i_means.append(int(mean))
            i_plussigmas.append(int(mean + sigma + 1))
            i_minussigmas.append(int(mean - sigma))
        bootstrap_indices = np.random.choice(np.arange(0, n), size=(n, n_bootstraps))
        bootstrap_indices.sort(axis=0)
        bootstrap_samples = data[bootstrap_indices]
        for i in range(len(i_means)):                
            try:    
                mean = bootstrap_samples[i_means[i], :]
                means.append(np.mean(mean))
                plussigma = bootstrap_samples[i_plussigmas[i], :]
                plussigmas.append(np.mean(plussigma))
                minussigma = bootstrap_samples[i_minussigmas[i], :]
                minussigmas.append(np.mean(minussigma))
            except IndexError:
                means.append(np.nan)
                plussigmas.append(np.nan)
                minussigmas.append(np.nan)
        return means, plussigmas, minussigmas

    def calculate_performance(self, values):
        means, plussigmas, minussigmas = self.estimate_percentile(values, [0.25, 0.75])
        e_quartiles = []
        e_quartiles.append((plussigmas[0] - minussigmas[0]) / 2)
        e_quartiles.append((plussigmas[1] - minussigmas[1]) / 2)
        sigma, e_sigma = self.convert_iqr_to_sigma(means, e_quartiles)
        if e_sigma != e_sigma:
            sigma = np.nan
        return sigma, e_sigma

    def plot_error_in_energy_bin(self, values, name, bin_no, energy_range):
        file_name = get_project_root().joinpath(
            'plots/error_distribution_' + name + '_' + str(bin_no) + '.pdf'
        )
        fig, ax = plt.subplots()
        ax.hist(values, bins='fd', density=True)
        ax.set(
            xlabel='Error',
            ylabel='Frequency',
            title='Network {} performance in energy bin {}'.format(name, energy_range)
        )
        fig.savefig(str(file_name))
        plt.close(fig)

    def create_comparison_plot(self, bins):
        for metric in self.config.comparison_metrics:
            print(
                'Starting {} metric comparison at {}'
                .format(
                    metric,
                    get_time()
                )
            )
            no_of_samples_in_bin = []
            bin_center = []
            opponent_performance = []
            opponent_std = []
            own_performance = []
            own_std = []
            width = []
            error_point_width = []
            for i in range(len(bins)):
                indexer = (
                    (self.comparison_df.binned == bins[i])
                    & (self.comparison_df.metric == metric)
                )
                no_of_samples_in_bin.append(len(self.comparison_df[indexer]))
                bin_center.append(bins[i].mid)
                width.append(bins[i].length)
                error_point_width.append(bins[i].length / 2)
                opponent = self.calculate_performance(
                    self.comparison_df[indexer].opponent_error.values
                )
                own = self.calculate_performance(
                    self.comparison_df[indexer].own_error.values
                )
                if metric == 'azimuth' or metric == 'zenith':
                    self.plot_error_in_energy_bin(
                        np.rad2deg(self.comparison_df[indexer].opponent_error),
                        metric,
                        i,
                        bins[i]
                    )
                    opponent_performance.append(np.rad2deg(opponent[0]))
                    opponent_std.append(np.rad2deg(opponent[1]))
                    own_performance.append(np.rad2deg(own[0]))
                    own_std.append(np.rad2deg(own[1]))
                elif metric == 'energy':
                    self.plot_error_in_energy_bin(
                        self.comparison_df[indexer].opponent_error,
                        metric,
                        i,
                        bins[i]
                    )
                    opponent_performance.append(opponent[0])
                    opponent_std.append(opponent[1])
                    own_performance.append(own[0])
                    own_std.append(own[1])
                elif metric == 'time':
                    self.plot_error_in_energy_bin(
                        self.comparison_df[indexer].opponent_error,
                        metric,
                        i,
                        bins[i]
                    )
                    opponent_performance.append(opponent[0])
                    opponent_std.append(opponent[1])
                    own_performance.append(own[0])
                    own_std.append(own[1])
            fig1, ax1 = plt.subplots()
            markers, caps, bars = ax1.errorbar(
                bin_center,
                opponent_performance,
                yerr=opponent_std,
                xerr=error_point_width,
                marker='x',
                ls='none',
                label='retro_crs'
            )
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            markers, caps, bars = ax1.errorbar(
                bin_center,
                own_performance,
                yerr=own_std,
                xerr=error_point_width,
                marker='x',
                ls='none',
                label='own'
            )
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            ax2 = ax1.twinx()
            ax2.hist(
                self.comparison_df.true_energy.values,
                bins=24,
                histtype='step'
            )
            ax2.set_yscale('log')
            ax1.set(xlabel='Log(E) [E/GeV]', title=metric)
            if self.config.comparison_type == 'azimuth' or self.config.comparison_type == 'zenith':
                ax1.set(ylabel='Error [Deg]')
            elif self.config.comparison_type == 'energy':
                ax1.set(ylabel='Relative error')
            elif self.config.comparison_type == 'time':
                ax1.set(ylabel='Error [ns]')
            ax2.set(ylabel='Events')
            ax1.legend()
            if self.config.wandb == True:
                buf = io.BytesIO()
                fig1.savefig(buf, format='png', dpi=600)
                buf.seek(0)
                im = Image.open(buf)
                self.wandb.log(
                    {
                        metric + ' performance': [
                            self.wandb.Image(im)
                        ]
                    }
                )
                buf.close()
            else:
                file_name = get_project_root().joinpath('plots/' + metric + '.pdf')
                fig1.savefig(str(file_name))


    def testing_ended(self):
        self.comparison_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.comparison_df.dropna(inplace=True)
        bins = self.calculate_energy_bins()
        self.create_comparison_plot(bins)
