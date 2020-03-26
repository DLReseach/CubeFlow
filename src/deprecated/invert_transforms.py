import numpy as np
import joblib


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

    def transform_inversion(self, data):
        if self.transformers is not None:
            for i, target in enumerate(self.config.targets):
                if target in list(self.transformers.keys()):
                    temp = self.transformers[target].inverse_transform(
                        np.array(data[target]).reshape(-1, 1)
                    ).flatten()
                    data[target] = temp
                    temp = self.transformers[target].inverse_transform(
                        np.array(data[target.replace('true', 'own')]).reshape(-1, 1)
                    ).flatten()
                    data[target.replace('true', 'own')] = temp
        return data
