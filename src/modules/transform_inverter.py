import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


class DomChargeScaler:
    def __init__(self):
        self._transformer = StandardScaler()
        self._clipmin = -np.inf
        self._clipmax = 100.0

    def transform(self, data):
        clipped_data = np.clip(data, self._clipmin, self._clipmax)
        data_transformed = self._transformer.transform(
            clipped_data.reshape(-1, 1)
        )
        return data_transformed

    def fit(self, data):
        clipped_data = np.clip(data, self._clipmin, self._clipmax)
        self._transformer.fit(clipped_data.reshape(-1, 1))


class EnergyNoLogTransformer:
    def transform(self, logE):
        E = np.power(10.0, logE)
        data_transformed = E / self._std
        return data_transformed

    def fit(self, logE):
        E = np.power(10.0, logE)
        self._std = np.std(E)


class InvertTransforms:
    def __init__(self, transformer_file):
        self.transformer_file = transformer_file
        with open(transformer_file, 'rb') as f:
            self.transformers = pickle.load(f)

    def invert_transform(self, values):
        for key in values:
            if self.transformers[key] is not None:
                values[key] = self.transformers[key].inverse_transform(
                    values[key].reshape(-1, 1)
                ).flatten()
        return values
