import joblib
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler


class TransformsInverter():
    def __init__(self, values, config, files_and_dirs):
        super().__init__()
        self.values = values
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.transformers = self.get_transformers()
 
    def get_transformers(self):
        transformer_file = self.files_and_dirs['transformer_dir'].joinpath(
            str(self.config.particle_type) + '_' + self.config.transform + '.pickle'
        )
        transformers = joblib.load(transformer_file)
        self.transformers = {}
        for target in self.config.targets:
            if target in transformers:
                self.transformers[target] = transformers[target]

    def invert_transform(self):
        if self.transformers is not None:
            for i, target in enumerate(self.config.targets):
                if target in self.transformers:
                    self.values[:, i] = self.transformers[target].inverse_transform(
                        self.values[:, i].reshape(-1, 1)
                    )
                    self.values[:, i] = torch.from_numpy(self.values[:, i])
        else:
            self.values = self.values
        return self.values
