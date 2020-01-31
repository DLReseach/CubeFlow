import torch
import torch.nn.functional as F
import joblib
import h5py as h5
import pandas as pd
import numpy as np
import math
from PIL import Image
import io
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from utils.math_funcs import unit_vector
from utils.utils import get_project_root
from utils.utils import get_time

import matplotlib.pyplot as plt


class RetroCrsComparison():
    def __init__(self, wandb, config):
        super().__init__()
        self.wandb = wandb
        self.config = config
        self.get_transformers()

        self.column_names = [
            'own_error',
            'opponent_error',
            'true_energy',
            'metric'
        ]
        self.comparison_df = pd.DataFrame(columns=self.column_names)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def get_transformers(self):
        root = get_project_root()
        transformer_file = root.joinpath(
            'data/' + self.config.data_type + '/transformers/'
            + str(self.config.particle_type) + '_' + self.config.transform
            + '.pickle'
        )
        transformers = joblib.load(transformer_file)
        self.transformers = {}
        for target in self.config.targets:
            if target in transformers:
                self.transformers[feature] = transformers[feature]


    def invert_transform(self, values):
        if self.transformers:
            for i, target in enumerate(self.transformers):
                values[:, i] = self.transformers[target].inverse_transform(
                    values[:, i]
                )
        return values


    def sort_values(self, idx, predictions, truth):
        sorted_idx = sorted(idx)
        sorted_predictions = torch.stack(
            [
                y for _, y in sorted(
                    zip(idx, predictions),
                    key=lambda pair: pair[0]
                )
            ]
        )
        sorted_truth = torch.stack(
            [
                y for _, y in sorted(
                    zip(idx, truth),
                    key=lambda pair: pair[0]
                )
            ]
        )
        return sorted_idx, sorted_predictions, sorted_truth


    def convert_to_spherical(self, values):
        values = self.invert_transform(values)
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
            ).float().to(self.device)
        elif angle_type == 'zenith':
            reversed_angles = torch.tensor(np.pi - angles).float().to(self.device)
        else:
            print('Unknown angle type')
        return reversed_angles


    def get_comparisons(self, file, idx):
        metrics = {}
        with h5.File(file, 'r') as f:
            for metric in self.config.comparison_metrics:
                dataset = 'raw/' + self.config.opponent + '_' + metric
                metrics[metric] = f[dataset][idx]
                # if metric == 'azimuth':
                    # print('Comparison azimuths:', metrics[metric])
                metrics[metric] = self.convert_to_signed_angle(
                    metrics[metric],
                    metric
                )
                # if metric == 'azimuth':
                    # print('Comparison reversed azimuths:', metrics[metric])
            dataset = 'raw/' + 'true_primary_energy'
            true_energy = torch.from_numpy(f[dataset][idx]).float().to(self.device)
        return metrics, true_energy


    def delta_angle(self, prediction, truth, angle_type, opponent_type):
        x = prediction
        y = truth
        difference = x - y
        if angle_type == 'zenith':
            delta = difference
        elif angle_type == 'azimuth':
            # delta = torch.atan2(torch.sin(difference), torch.cos(difference))
            # delta = ((difference + np.pi) % (2 * np.pi)) - np.pi
            delta = torch.where(
                abs(difference) > np.pi,
                - 2 * torch.sign(difference) * np.pi + difference,
                difference
            )
            # if opponent_type == 'opponent':
                # print('Truth:', y)
                # print('Prediction:', x)
                # print('Delta angle:', delta)
        return delta


    def update_values(self, predictions, truth, comparisons, energy):
        opponent_error = {}
        own_error = {}
        converted_predictions = self.convert_to_spherical(predictions)
        converted_truth = self.convert_to_spherical(truth)
        for i, metric in enumerate(self.config.comparison_metrics):
            normalized_comparisons = self.convert_to_signed_angle(
                comparisons[i],
                metric
            )
            opponent_error[metric] = self.delta_angle(
                normalized_comparisons,
                converted_truth[metric],
                metric,
                'opponent'
            )
            own_error[metric] = self.delta_angle(
                converted_predictions,
                converted_truth[metric],
                metric,
                'own'
            )
        for metric in self.config.comparison_metrics:
            temp_df = pd.DataFrame(
                data={
                    'own_error': own_error[metric].tolist(),
                    'opponent_error': opponent_error[metric].tolist(),
                    'true_energy': true_energy.tolist(),
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


    def bootstrap(self, series, R=1000):
        n = series.size
        column_names = [
            'q1',
            'q1_minus',
            'q1_plus',
            'q3',
            'q3_minus',
            'q3_plus'
        ]
        df = pd.DataFrame(columns=column_names)
        for i in range(R):
            bs_sample = series.sample(n=n, replace=True)
            temp_df = pd.DataFrame(
                data={
                    'q1': [bs_sample.quantile(q=0.25)],
                    'q1_minus': [bs_sample.quantile(q=0.21)],
                    'q1_plus': [bs_sample.quantile(q=0.29)],
                    'q3': [bs_sample.quantile(q=0.75)],
                    'q3_minus': [bs_sample.quantile(q=0.71)],
                    'q3_plus': [bs_sample.quantile(q=0.79)]
                }
            )
            df = df.append(
                temp_df,
                ignore_index=True
            )
        q1_mean = df.q1.mean()
        q3_mean = df.q3.mean()
        resolution = (q3_mean - q1_mean) / 1.35
        resolution_error = np.sqrt(
            (1 / 1.35)**2 * (df.q3_plus.mean())**2
            + (- 1 / 1.35)**2 * (df.q1_plus.mean())**2
        )
        return [resolution, resolution_error]


    def plot_error_in_energy_bin(self, values, name, bin_no):
        file_name = get_project_root().joinpath(
            'plots/error_distribution_' + name + '_' + str(bin_no) + '.pdf'
        )
        fig, ax = plt.subplots()
        ax.hist(values, bins='fd')
        fig.savefig(str(file_name))


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
            for i in range(len(bins)):
                indexer = (
                    (self.comparison_df.binned == bins[i])
                    & (self.comparison_df.metric == metric)
                )
                no_of_samples_in_bin.append(len(self.comparison_df[indexer]))
                bin_center.append(bins[i].mid)
                width.append(bins[i].length)
                opponent = self.bootstrap(
                    self.comparison_df[indexer].opponent_error
                )
                opponent_performance.append(opponent[0])
                opponent_std.append(opponent[1])
                own = self.bootstrap(
                    self.comparison_df[indexer].own_error
                )
                own_performance.append(own[0])
                own_std.append(own[1])
                self.plot_error_in_energy_bin(
                    self.comparison_df[indexer].opponent_error,
                    metric,
                    i
                )
            fig1, ax1 = plt.subplots()
            ax1.bar(
                bin_center,
                no_of_samples_in_bin,
                align='center',
                fill=False,
                width=width,
                edgecolor='black'
            )
            ax2 = ax1.twinx()
            markers, caps, bars = ax2.errorbar(
                bin_center,
                opponent_performance,
                yerr=opponent_std,
                # xerr=width,
                fmt='o',
                ecolor='black',
                capsize=2,
                capthick=2,
                label='retro_crs',
                markerfacecolor='blue'
            )
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            markers, caps, bars = ax2.errorbar(
                bin_center,
                own_performance,
                yerr=own_std,
                # xerr=width,
                fmt='o',
                ecolor='black',
                capsize=2,
                capthick=2,
                label='own',
                markerfacecolor='green'
            )
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
            ax1.set_yscale('log')
            ax1.set(xlabel='Energy', ylabel='Frequency', title=metric)
            ax2.set(ylabel='Error')
            ax2.legend()
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
        bins = self.calculate_energy_bins()
        self.create_comparison_plot(bins)
