import h5py as h5
import numpy as np


class DistributionHistograms:
    def __init__(self, train, validation, test, config):
        self.train = train
        self.validation = validation
        self.test = test
        self.config = config


    def create_histograms(self, set_df):
        files = set_df.file.unique()
        histogram_dict = {}
        for feature in self.config.features:
            histogram_dict[feature] = []
        for target in self.config.targets:
            histogram_dict[target] = []
        for file in files:
            idx = sorted(set_df[set_df.file == file].idx.values.tolist())
            with h5.File(file, 'r') as f:
                for feature in self.config.features:
                    masks = f['masks/' + self.config.mask][idx]
                    dataset = self.config.transform + '/' + feature
                    if dataset in f:
                        data = f[dataset][idx]
                    else:
                        data = f['raw/' + feature][idx]
                    for i in range(len(data)):
                        histogram_dict[feature].extend(list(data[i][masks[i]]))
                for target in self.config.targets:
                    dataset = self.config.transform + '/' + target
                    if dataset in f:
                        data = f[dataset][idx]
                    else:
                        data = f['raw/' + target][idx]
                    for i in range(len(data)):
                        histogram_dict[target].append(data[i])
        return histogram_dict