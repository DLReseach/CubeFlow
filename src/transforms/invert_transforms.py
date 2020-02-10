import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler


class TransformsInverter():
    def __init__(self, y, y_hat, config, files_and_dirs):
        super().__init__()
        self.y = y
        self.y_hat = y_hat
        self.config = config
        self.files_and_dirs = files_and_dirs
 
    def transform_inversion(self):
        transformer_file = self.files_and_dirs['transformer_dir'].joinpath(
            str(self.config.particle_type) + '_' + self.config.transform + '.pickle'
        )
        transformers = joblib.load(transformer_file)
        self.transformers = {}
        for target in self.config.targets:
            if target in transformers:
                self.transformers[target] = transformers[target]
        if self.transformers is not None:
            for i, target in enumerate(self.config.targets):
                if target in list(self.transformers.keys()):
                    temp = self.transformers[target].inverse_transform(
                        self.y[:, i].reshape(-1, 1)
                    ).flatten()
                    self.y[:, i] = torch.from_numpy(temp)
                    temp = self.transformers[target].inverse_transform(
                        self.y_hat[:, i].numpy().reshape(-1, 1)
                    ).flatten()
                    self.y_hat[:, i] = torch.from_numpy(temp)
        return self.y, self.y_hat
