import torch
import torch.nn.functional as F
import joblib
import h5py as h5
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from utils.math_funcs import unit_vector
from utils.utils import get_project_root

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
        phi = torch.atan(y / x)
        theta = torch.acos(z / r)
        return {'azimuth': phi, 'zenith': theta}


    def get_comparisons(self, file, idx):
        metrics = {}
        with h5.File(file, 'r') as f:
            for metric in self.config.comparison_metrics:
                dataset = 'raw/' + self.config.opponent + '_' + metric
                metrics[metric] = torch.from_numpy(f[dataset][idx])
            dataset = 'raw/' + 'true_primary_energy'
            true_energy = torch.from_numpy(f[dataset][idx])
        return metrics, true_energy


    def calculate_errors(self, prediction, truth):
        error = prediction.sub(truth)
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
        own_errors = self.comparison_df[
            (self.comparison_df.binned == bins[0])
            & (self.comparison_df.metric == 'azimuth')
        ].own_error.values
        self.plot_error_in_energy_bin(own_errors)
        # retro_azimuth = []
        # predicted_azimuth = []
        # retro_zenith = []
        # predicted_zenith = []
        # for i in range(len(bins)):
        #     retro = ((self.comparison_df[self.comparison_df.binned == bins[i]].retro_crs_azimuth.mean()) * 180 / np.pi) % 360
        #     predict = ((self.comparison_df[self.comparison_df.binned == bins[i]].predicted_azimuth.mean()) * 180 / np.pi) % 360
        #     retro_azimuth.append(retro)
        #     predicted_azimuth.append(predict)
        #     retro = ((self.comparison_df[self.comparison_df.binned == bins[i]].retro_crs_zenith.mean()) * 180 / np.pi) % 360
        #     predict = ((self.comparison_df[self.comparison_df.binned == bins[i]].predicted_zenith.mean()) * 180 / np.pi) % 360
        #     retro_zenith.append(retro)
        #     predicted_zenith.append(predict)
        # hist, bins = np.histogram(
        #     self.comparison_df.true_energy.values,
        #     bins=10
        # )
        # width = 0.7 * (bins[1] - bins[0])
        # center = (bins[:-1] + bins[1:]) / 2
        # fig1, ax1 = plt.subplots()
        # ax1.bar(
        #     center,
        #     hist,
        #     align='center',
        #     width=width
        # )
        # ax2 = ax1.twinx()
        # ax2.scatter(
        #     center,
        #     retro_azimuth
        # )
        # ax2.scatter(
        #     center,
        #     predicted_azimuth
        # )
        # ax1.set_yscale('log')
        # ax1.set(xlabel='Energy', ylabel='Frequency', title='Azimuth')
        # ax2.set(ylabel='Error')
        # fig1.savefig(str(get_project_root().joinpath('azimuth.pdf')))
        # fig2, ax3 = plt.subplots()
        # ax3.bar(
        #     center,
        #     hist,
        #     align='center',
        #     width=width
        # )
        # ax4 = ax3.twinx()
        # ax4.scatter(
        #     center,
        #     retro_zenith
        # )
        # ax4.scatter(
        #     center,
        #     predicted_zenith
        # )
        # ax3.set(xlabel='Energy', ylabel='Frequency', title='Zenith')
        # ax4.set(ylabel='Error')
        # fig2.savefig(str(get_project_root().joinpath('zenith.pdf')))
        # self.wandb.log({'Azimuth error': fig1})
        # self.wandb.log({'Zenith error': fig2})


    def plot_error_in_energy_bin(self, values):
        file_name = get_project_root().joinpath('plots/error_distribution.pdf')
        fig, ax = plt.subplots()
        ax.hist(values, bins='auto')
        fig.savefig(str(file_name))

