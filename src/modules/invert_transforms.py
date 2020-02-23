import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler


class TransformsInverter():
    def __init__(self, config, files_and_dirs):
        super().__init__()
        self.config = config
        self.files_and_dirs = files_and_dirs
        transformer_file = self.files_and_dirs['transformer_dir'].joinpath(
            str(self.config.particle_type) + '_' + self.config.transform + '.pickle'
        )
        transformers = joblib.load(transformer_file)
        self.transformers = {}
        for target in self.config.targets:
            if target in transformers:
                self.transformers[target] = transformers[target]

    def transform_inversion(self, y, y_hat):
        if self.transformers is not None:
            for i, target in enumerate(self.config.targets):
                if target in list(self.transformers.keys()):
                    temp = self.transformers[target].inverse_transform(
                        y[:, i].cpu().reshape(-1, 1)
                    ).flatten()
                    y[:, i] = torch.from_numpy(temp)
                    temp = self.transformers[target].inverse_transform(
                        y_hat[:, i].cpu().reshape(-1, 1)
                    ).flatten()
                    y_hat[:, i] = torch.from_numpy(temp)
        return y, y_hat
