import torch
import joblib
import h5py as h5
import pandas as pd
import numpy as np
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
        self.column_names = [
            'file',
            'idx',
            'true_energy',
            'predicted_azimuth',
            'predicted_zenith',
            'retro_crs_azimuth',
            'retro_crs_zenith'
        ]
        self.comparison_df = pd.DataFrame(columns=self.column_names)
        # self.predicted_angles = []
        # self.retro_crs_angles = []
        # self.true_energy = []
        self.get_transformers()


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
                values[i] = self.transformers[target].inverse_transform(
                    values[i]
                )
        return values


    def convert_to_spherical(self, values):
        values = self.invert_transform(values)
        unit_vec = unit_vector(values)
        x = unit_vec[0]
        y = unit_vec[1]
        z = unit_vec[2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        phi = torch.atan(y / x)
        theta = torch.acos(z / r)
        return phi.item(), theta.item()


    def get_retro_crs_angles_and_true_energy(self, file, idx):
        with h5.File(file, 'r') as f:
            azimuth = f['raw/retro_crs_prefit_azimuth'][idx]
            zenith = f['raw/retro_crs_prefit_zenith'][idx]
            energy = f[self.config.transform + '/true_primary_energy'][idx]
        return azimuth, zenith, energy


    def update_values(self, file, idx, predictions):
        for i in range(predictions.size(0)):
            predicted_phi, predicted_theta = self.convert_to_spherical(
                predictions[i, :]
            )
            retro_crs_azimuth, retro_crs_zenith, true_energy =\
                self.get_retro_crs_angles_and_true_energy(
                    file, idx[i]
                )
            temp_df = pd.DataFrame(
                data=[
                    [
                        file,
                        idx[i],
                        true_energy,
                        predicted_phi,
                        predicted_theta,
                        retro_crs_azimuth,
                        retro_crs_zenith
                    ]
                ],
                columns=self.column_names
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
        retro_azimuth = []
        predicted_azimuth = []
        for i in range(len(bins)):
            retro = self.comparison_df[self.comparison_df.binned == bins[i]].retro_crs_azimuth.mean()
            predict = self.comparison_df[self.comparison_df.binned == bins[i]].predicted_azimuth.mean()
            retro_azimuth.append(retro)
            predicted_azimuth.append(predict)
        hist, bins = np.histogram(
            self.comparison_df.true_energy.values,
            bins=10
        )
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        fig, ax1 = plt.subplots()
        ax1.bar(
            center,
            hist,
            align='center',
            width=width
        )
        ax2 = ax1.twinx()
        ax2.scatter(
            center,
            retro_azimuth
        )
        ax2.scatter(
            center,
            predicted_azimuth
        )
        fig.savefig(str(get_project_root().joinpath('energy_test.pdf')))
