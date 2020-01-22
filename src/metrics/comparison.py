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
        values = F.normalize(values)
        x = values[:, 0]
        y = values[:, 1]
        z = values[:, 2]
        hypot = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(z, hypot) % 2 * np.pi
        phi = torch.atan2(y, x) % 2 * np.pi
        # theta = np.rad2deg(theta) % 360
        # phi = np.rad2deg(phi) % 360
        return {'azimuth': phi, 'zenith': theta}


    def get_comparisons(self, file, idx):
        metrics = {}
        with h5.File(file, 'r') as f:
            for metric in self.config.comparison_metrics:
                dataset = 'raw/' + self.config.opponent + '_' + metric
                metrics[metric] = torch.from_numpy(f[dataset][idx]).float().to(self.device)
                # metrics[metric] = np.rad2deg(metrics[metric])
            dataset = 'raw/' + 'true_primary_energy'
            true_energy = torch.from_numpy(f[dataset][idx]).float().to(self.device)
        return metrics, true_energy


    def calculate_errors(self, prediction, truth):
        # print(prediction)
        # print(truth)
        # error = prediction.sub(truth)
        # print('Prediction:', prediction)
        # print('Truth:', truth)
        test = prediction - truth
        # print('Error 1:', test)
        # error = ((error + np.pi) % 2 * np.pi) - np.pi
        error = torch.atan2(
            torch.sin(prediction - truth),
            torch.cos(prediction - truth)
        )
        print('Error 2:', error)
        return error


    def update_values(self, file, idx, predictions, truth):
        assert type(file) == str, 'Hmm, file is not a string'
        opponent_error = {}
        own_error = {}
        sorted_idx, sorted_predictions, sorted_truth = self.sort_values(
            idx,
            predictions,
            truth
        )
        sorted_predictions = self.invert_transform(sorted_predictions)
        sorted_truth = self.invert_transform(sorted_truth)
        sorted_predictions = self.convert_to_spherical(sorted_predictions)
        sorted_truth = self.convert_to_spherical(sorted_truth)
        comparison_metrics, true_energy = self.get_comparisons(file, sorted_idx)
        for metric in self.config.comparison_metrics:
            opponent_error[metric] = self.calculate_errors(
                comparison_metrics[metric],
                sorted_truth[metric]
            )
            own_error[metric] = self.calculate_errors(
                sorted_predictions[metric],
                sorted_truth[metric]
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
        no_of_bins = 10
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
        print(df)
        q1_mean = df.q1.mean()
        q3_mean = df.q3.mean()
        std = (q3_mean - q1_mean) / 1.35
        return std


    def plot_error_in_energy_bin(self, values):
        file_name = get_project_root().joinpath('plots/error_distribution.pdf')
        fig, ax = plt.subplots()
        ax.hist(values, bins='auto')
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
                if i == 2:
                    self.plot_error_in_energy_bin(self.comparison_df[indexer].opponent_error)
                indexer = (
                    (self.comparison_df.binned == bins[i])
                    & (self.comparison_df.metric == metric)
                )
                no_of_samples_in_bin.append(len(self.comparison_df[indexer]))
                bin_center.append(bins[i].mid)
                width.append(bins[i].length)
                opponent_bs_std = self.bootstrap(
                    self.comparison_df[indexer].opponent_error
                )
                opponent_performance.append(
                    abs(
                        self.comparison_df[indexer].opponent_error.mean()
                    )
                )
                opponent_std.append(opponent_bs_std)
                own_bs_std = self.bootstrap(
                    self.comparison_df[indexer].own_error
                )
                own_performance.append(
                    abs(
                        self.comparison_df[indexer].own_error.mean()
                    )
                )
                own_std.append(own_bs_std)
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
                # marker='o',
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
                # marker='o',
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
