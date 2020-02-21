import torch
import torch.nn.functional as F
from pathlib import Path
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
plt.rc('text', usetex=True)


class ResolutionComparison():
    def __init__(self, wandb, config):
        super().__init__()
        self.wandb = wandb
        self.config = config
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
        get_project_root().joinpath('temp').mkdir(exist_ok=True)

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
                matched_metrics[comparison]['opponent'] = normalized_comparisons.to(self.device)
            elif comparison == 'energy':
                needed_targets = [
                    'true_primary_energy'
                ]
                needed_targets_test = all(x in self.config.targets for x in needed_targets)
                assert needed_targets_test, 'Targets missing for {} comparison'.format(comparison)
                target_indices = []
                for target in needed_targets:
                    target_indices.append(self.config.targets.index(target))
                matched_metrics[comparison] = {}
                matched_metrics[comparison]['own'] = predictions[:, target_indices].flatten()
                matched_metrics[comparison]['truth'] = truth[:, target_indices].flatten()
                matched_metrics[comparison]['opponent'] = torch.log10(comparisons[comparison])
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

    def calculate_energy_bins(self):
        no_of_bins = 24
        self.comparison_df['energy_binned'] = pd.cut(
            self.comparison_df['true_energy'], no_of_bins
        )
        bins = self.comparison_df.energy_binned.unique()
        bins.sort_values(inplace=True)
        return bins

    def calculate_dom_bins(self):
        no_of_bins = 20
        self.comparison_df['dom_binned'] = pd.cut(
            self.comparison_df['event_length'], no_of_bins
        )
        bins = self.comparison_df.dom_binned.unique()
        bins.sort_values(inplace=True)
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
        means, plussigmas, minussigmas = self.estimate_percentile(values, [0.16, 0.84])
        e_quartiles = []
        e_quartiles.append((plussigmas[0] - minussigmas[0]) / 2)
        e_quartiles.append((plussigmas[1] - minussigmas[1]) / 2)
        sigma, e_sigma = self.convert_iqr_to_sigma(means, e_quartiles)
        if e_sigma != e_sigma:
            sigma = np.nan
        return sigma, e_sigma

    def plot_error_in_energy_bin(self, own, opponent, name, bin_no, energy_range):
        file_name = get_project_root().joinpath(
            'temp/error_distribution_' + name + '_' + str(bin_no) + '.png'
        )
        fig, ax = plt.subplots()
        ax.hist(own, bins='fd', density=True, alpha=0.5, label='CubeFlow')
        ax.hist(opponent, bins='fd', density=True, alpha=0.5, label='IceCube')
        ax.set(
            xlabel='Error',
            ylabel='Frequency',
            title='Network {} performance in energy bin {}'.format(name, energy_range)
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(file_name))
        if self.config.wandb:
            self.wandb.save(str(file_name))
        plt.close(fig)

    def create_comparison_plot(self, bins, bin_type):
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
            for i, bin_no in enumerate(bins):
                if bin_type == 'energy':
                    indexer = (
                        (self.comparison_df.energy_binned == bin_no)
                        & (self.comparison_df.metric == metric)
                    )
                elif bin_type == 'doms':
                    indexer = (
                        (self.comparison_df.dom_binned == bin_no)
                        & (self.comparison_df.metric == metric)
                    )
                no_of_samples_in_bin.append(len(self.comparison_df[indexer]))
                bin_center.append(bin_no.mid)
                width.append(bin_no.length)
                error_point_width.append(bin_no.length / 2)
                opponent = self.calculate_performance(
                    self.comparison_df[indexer].opponent_error.values
                )
                own = self.calculate_performance(
                    self.comparison_df[indexer].own_error.values
                )
                if metric == 'azimuth' or metric == 'zenith':
                    self.plot_error_in_energy_bin(
                        np.rad2deg(self.comparison_df[indexer].own_error),
                        np.rad2deg(self.comparison_df[indexer].opponent_error),
                        metric,
                        i,
                        bin_no
                    )
                    opponent_performance.append(np.rad2deg(opponent[0]))
                    opponent_std.append(np.rad2deg(opponent[1]))
                    own_performance.append(np.rad2deg(own[0]))
                    own_std.append(np.rad2deg(own[1]))
                elif metric == 'energy':
                    self.plot_error_in_energy_bin(
                        self.comparison_df[indexer].own_error,
                        self.comparison_df[indexer].opponent_error,
                        metric,
                        i,
                        bin_no
                    )
                    opponent_performance.append(opponent[0])
                    opponent_std.append(opponent[1])
                    own_performance.append(own[0])
                    own_std.append(own[1])
                elif metric == 'time':
                    self.plot_error_in_energy_bin(
                        self.comparison_df[indexer].own_error,
                        self.comparison_df[indexer].opponent_error,
                        metric,
                        i,
                        bin_no
                    )
                    opponent_performance.append(opponent[0])
                    opponent_std.append(opponent[1])
                    own_performance.append(own[0])
                    own_std.append(own[1])
            fig1, (reso_ax, ratio_ax) = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={
                    'height_ratios': [3, 1]
                }
            )
            reso_ax.xaxis.set_ticks_position('none') 
            markers, caps, bars = reso_ax.errorbar(
                bin_center,
                own_performance,
                yerr=own_std,
                xerr=error_point_width,
                marker='.',
                markersize=1,
                ls='none',
                label=r'$\mathrm{CubeFlow}$'
            )
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            markers, caps, bars = reso_ax.errorbar(
                bin_center,
                opponent_performance,
                yerr=opponent_std,
                xerr=error_point_width,
                marker='.',
                markersize=1,
                ls='none',
                label=r'$\mathrm{IceCube}$'
            )
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            rel_improvement = np.divide(np.array(own_performance) - np.array(opponent_performance), np.array(opponent_performance))
            ratio_ax.plot(bin_center, rel_improvement, '.')
            ratio_ax.set(ylabel=r'$\mathrm{Rel. \ imp.}$')
            hist_ax = reso_ax.twinx()
            if bin_type == 'energy':
                hist_ax.hist(
                    self.comparison_df.true_energy.values,
                    bins=len(bins),
                    histtype='step',
                    color='grey'
                )
            elif bin_type == 'doms':
                hist_ax.hist(
                    self.comparison_df.event_length.values,
                    bins=len(bins),
                    histtype='step',
                    color='grey'
                )
            hist_ax.set_yscale('symlog')
            hist_ax.set_ylim(ymin=0)
            if metric == 'energy':
                reso_ax.set_ylim(ymin=0, ymax=2)
            else:
                reso_ax.set_ylim(ymin=0)
            if bin_type == 'energy':
                ratio_ax.set(xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$')
            elif bin_type == 'doms':
                ratio_ax.set(xlabel=r'$\mathrm{No. \ of \ DOMs}$')
            ratio_ax.axhline(y=0, linestyle='dashed')
            if metric == 'azimuth':
                reso_ax.set(
                    title=r'$\mathrm{Azimuth \ reconstruction \ comparison}$',
                    ylabel=r'$\mathrm{Resolution} \; [\mathrm{Deg}]$'
                )
            if metric == 'zenith':
                reso_ax.set(
                    title=r'$\mathrm{Zenith \ reconstruction \ comparison}$',
                    ylabel=r'$\mathrm{Resolution} \; [\mathrm{Deg}]$'
                )
            elif metric == 'energy':
                reso_ax.set(
                    title=r'$\mathrm{Energy \ reconstruction \ comparison}$',
                    ylabel=r'$\left( \log_{10}{E} - \log_{10}{E_{\mathrm{true}}} \right) / \log_{10}{E_{\mathrm{true}}} \; [\%]$'
                )
            elif metric == 'time':
                reso_ax.set(
                    title=r'$\mathrm{Time \ reconstruction \ comparison}$',
                    ylabel=r'$\mathrm{Resolution} \; [\mathrm{ns}]$')
            hist_ax.set(ylabel=r'$\mathrm{Events}$')
            reso_ax.legend()
            fig1.tight_layout()
            if self.config.wandb == True and bin_type == 'energy':
                buf = io.BytesIO()
                fig1.savefig(buf, format='png', dpi=600)
                buf.seek(0)
                im = Image.open(buf)
                self.wandb.log(
                    {
                        bin_type + metric + ' performance': [
                            self.wandb.Image(im)
                        ]
                    }
                )
                buf.close()
            file_name = get_project_root().joinpath('temp/' + bin_type + '_' + metric + '.png')
            fig1.savefig(str(file_name))
            if self.config.wandb and bin_type == 'doms':
                self.wandb.save(file_name)
            plt.close(fig1)

    def icecube_2d_histogram(self, bins):
        for metric in self.config.comparison_metrics:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16.0, 6.0))
            indexer = self.comparison_df.metric == metric
            x_values = self.comparison_df[indexer].true_energy.values
            y_values_own = self.comparison_df[indexer].own_error.values
            _, x_bin_edges = np.histogram(x_values, bins='fd', range=[0, 3])
            if metric == 'azimuth':
                _, y_bin_edges_own = np.histogram(y_values_own, bins='fd', range=[-2.5, 2.5])
            if metric == 'energy':
                _, y_bin_edges_own = np.histogram(y_values_own, bins='fd', range=[-1, 4])
            if metric == 'time':
                _, y_bin_edges_own = np.histogram(y_values_own, bins='fd', range=[-150, 250])
            if metric == 'zenith':
                _, y_bin_edges_own = np.histogram(y_values_own, bins='fd', range=[-2, 2])
            widths1_own = np.linspace(min(x_bin_edges), max(x_bin_edges), int(0.5 + x_bin_edges.shape[0]/4.0))
            widths2_own = np.linspace(min(y_bin_edges_own), max(y_bin_edges_own), int(0.5 + y_bin_edges_own.shape[0]/4.0))
            low_own = []
            medians_own = []
            high_own = []
            centers_own = []
            for bin in sorted(bins):
                low_own.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].own_error.quantile(0.16))
                low_own.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].own_error.quantile(0.16))
                medians_own.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].own_error.quantile(0.5))
                medians_own.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].own_error.quantile(0.5))
                high_own.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].own_error.quantile(0.84))
                high_own.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].own_error.quantile(0.84))
                centers_own.append(bin.left)
                centers_own.append(bin.right)
            y_values_opponent = self.comparison_df[indexer].opponent_error.values
            if metric == 'azimuth':
                _, y_bin_edges_opponent = np.histogram(y_values_opponent, bins='fd', range=[-2.5, 2.5])
            if metric == 'energy':
                _, y_bin_edges_opponent = np.histogram(y_values_opponent, bins='fd', range=[-1, 4])
            if metric == 'time':
                _, y_bin_edges_opponent = np.histogram(y_values_opponent, bins='fd', range=[-150, 250])
            if metric == 'zenith':
                _, y_bin_edges_opponent = np.histogram(y_values_opponent, bins='fd', range=[-2, 2])
            widths1_opponent = np.linspace(min(x_bin_edges), max(x_bin_edges), int(0.5 + x_bin_edges.shape[0]/4.0))
            widths2_opponent = np.linspace(min(y_bin_edges_opponent), max(y_bin_edges_opponent), int(0.5 + y_bin_edges_opponent.shape[0]/4.0))
            low_opponent = []
            medians_opponent = []
            high_opponent = []
            centers_opponent = []
            for bin in sorted(bins):
                low_opponent.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].opponent_error.quantile(0.16))
                low_opponent.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].opponent_error.quantile(0.16))
                medians_opponent.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].opponent_error.quantile(0.5))
                medians_opponent.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].opponent_error.quantile(0.5))
                high_opponent.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].opponent_error.quantile(0.84))
                high_opponent.append(self.comparison_df[(indexer) & (self.comparison_df['energy_binned'] == bin)].opponent_error.quantile(0.84))
                centers_opponent.append(bin.left)
                centers_opponent.append(bin.right)
            counts_own, xedges_own, yedges_own, im_own = ax1.hist2d(
                self.comparison_df[indexer].true_energy.values,
                self.comparison_df[indexer].own_error.values,
                bins=[widths1_own, widths2_own],
                cmap='Oranges'
            )
            ax1.plot(centers_own, medians_own, linestyle='solid', color='red', alpha=0.5, label=r'$50 \% \ (\mathrm{CubeFlow})$')
            ax1.plot(centers_own, low_own, linestyle='dashed', color='red', alpha=0.5, label=r'$16 \% \ (\mathrm{CubeFlow})$')
            ax1.plot(centers_own, high_own, linestyle='dotted', color='red', alpha=0.5, label=r'$84 \% \ (\mathrm{CubeFlow})$')
            ax1.plot(centers_opponent, medians_opponent, linestyle='solid', color='green', alpha=0.5, label=r'$50 \% \ (\mathrm{IceCube})$')
            ax1.plot(centers_opponent, low_opponent, linestyle='dashed', color='green', alpha=0.5, label=r'$16 \% \ (\mathrm{IceCube})$')
            ax1.plot(centers_opponent, high_opponent, linestyle='dotted', color='green', alpha=0.5, label=r'$84 \% \ (\mathrm{IceCube})$')
            fig.colorbar(im_own, ax=ax1)
            ax1.set(xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$')
            if metric == 'energy':
                ax1.set(
                    title=r'$\mathrm{Energy \ reconstruction \ results \ CubeFlow}$',
                    ylabel=r'$\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}} \; [\%]$'
                )
            elif metric == 'azimuth':
                ax1.set(
                    title=r'$\mathrm{Azimuth \ reconstruction \ results \ CubeFlow}$',
                    ylabel=r'$\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}} \; [\mathrm{deg}]$'
                )
            elif metric == 'zenith':
                ax1.set(
                    title=r'$\mathrm{Zenith \ reconstruction \ results \ CubeFlow}$',
                    ylabel=r'$\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}} \; [\mathrm{deg}]$'
                )
            elif metric == 'time':
                ax1.set(
                    title=r'$\mathrm{Time \ reconstruction \ results \ Cubeflow}$',
                    ylabel=r'$t_{\mathrm{reco}} - t_{\mathrm{true}} \; [\mathrm{ns}]$'
                )
            ax1.legend()
            counts_opponent, xedges_opponent, yedges_opponent, im_opponent = ax2.hist2d(
                self.comparison_df[indexer].true_energy.values,
                self.comparison_df[indexer].opponent_error.values,
                bins=[widths1_opponent, widths2_opponent],
                cmap='Oranges'
            )
            ax2.plot(centers_own, medians_own, linestyle='solid', color='red', alpha=0.5, label=r'$50 \% \ (\mathrm{CubeFlow})$')
            ax2.plot(centers_own, low_own, linestyle='dashed', color='red', alpha=0.5, label=r'$16 \% \ (\mathrm{CubeFlow})$')
            ax2.plot(centers_own, high_own, linestyle='dotted', color='red', alpha=0.5, label=r'$84 \% \ (\mathrm{CubeFlow})$')
            ax2.plot(centers_opponent, medians_opponent, linestyle='solid', alpha=0.5, color='green', label=r'$50 \% \ (\mathrm{IceCube})$')
            ax2.plot(centers_opponent, low_opponent, linestyle='dashed', alpha=0.5, color='green', label=r'$16 \% \ (\mathrm{IceCube})$')
            ax2.plot(centers_opponent, high_opponent, linestyle='dotted', alpha=0.5, color='green', label=r'$84 \% \ (\mathrm{IceCube})$')
            fig.colorbar(im_opponent, ax=ax2)
            ax2.set(xlabel=r'$\log{E_{\mathrm{true}}} \; [E/\mathrm{GeV}]$')
            if metric == 'energy':
                ax2.set(
                    title=r'$\mathrm{Energy \ reconstruction \ results \ IceCube}$',
                    ylabel=r'$\frac{\log_{10}{E_{\mathrm{reco}}} - \log_{10}{E_{\mathrm{true}}}}{\log_{10}{E_{\mathrm{true}}}} \; [\%]$'
                )
            elif metric == 'azimuth':
                ax2.set(
                    title=r'$\mathrm{Azimuth \ reconstruction \ results \ IceCube}$',
                    ylabel=r'$\theta_{\mathrm{azimuth,reco}} - \theta_{\mathrm{azimuth,true}} \; [\mathrm{deg}]$'
                )
            elif metric == 'zenith':
                ax2.set(
                    title=r'$\mathrm{Zenith \ reconstruction \ results \ IceCube}$',
                    ylabel=r'$\theta_{\mathrm{zenith,reco}} - \theta_{\mathrm{zenith,true}} \; [\mathrm{deg}]$'
                )
            elif metric == 'time':
                ax2.set(
                    title=r'$\mathrm{Time \ reconstruction \ results \ IceCube}$',
                    ylabel=r'$t_{\mathrm{reco}} - t_{\mathrm{true}} \; [\mathrm{ns}]$'
                )
            ax2.legend()
            fig.tight_layout()
            if self.config.wandb == True:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=600)
                buf.seek(0)
                im = Image.open(buf)
                self.wandb.log(
                    {
                        metric + ' ic_performance': [
                            self.wandb.Image(im)
                        ]
                    }
                )
                buf.close()
            else:
                file_name = get_project_root().joinpath('temp/' + metric + '_ic.png')
                fig.savefig(str(file_name))
            plt.close(fig)

    def testing_ended(self):
        self.comparison_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.comparison_df.to_pickle('comparison_dataframe.gzip')
        if self.config.wandb:
            self.wandb.save('comparison_dataframe.csv')
        self.comparison_df.dropna(inplace=True)
        energy_bins = self.calculate_energy_bins()
        dom_bins = self.calculate_dom_bins()
        self.create_comparison_plot(energy_bins, 'energy')
        self.create_comparison_plot(dom_bins, 'doms')
        self.icecube_2d_histogram(energy_bins)
