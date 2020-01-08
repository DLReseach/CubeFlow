from torch.utils.data import Dataset
import numpy as np
import h5py as h5
import pandas as pd


class CnnGenerator(Dataset):
    def __init__(self, config, ids, test):
        self.config = config
        self.ids = ids
        self.test = test
        if self.config.dev_run == True:
            self.config.batch_size = 2
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(len(self.ids) / self.config.batch_size))


    def __getitem__(self, index):
        indices = self.indices[
            index * self.config.batch_size:(index + 1) * self.config.batch_size
            ]
        ids_temp = self.ids.iloc[indices]
        X, y = self.__data_generation(ids_temp)
        return X, y


    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        if self.config.shuffle == True and self.test == False:
            np.random.shuffle(self.indices)
        self.ids = self.ids.iloc[self.indices]
        self.ids.sort_values(by='file', inplace=True)
        self.indices = np.arange(len(self.ids))


    def __data_generation(self, ids_temp):
        X = np.zeros(
            (
                self.config.batch_size,
                self.config.max_doms,
                len(self.config.features)
            )
        )
        y = np.zeros((self.config.batch_size, len(self.config.targets)))
        files_dict = {
            file_name: list(ids_temp[ids_temp.file == file_name].idx.values)
            for file_name in ids_temp.file.unique()
        }
        batch_size_dim_start = 0
        for file, idx in files_dict.items():
            idx = sorted(idx)
            batch_size_dim_end = batch_size_dim_start + len(idx)
            with h5.File(file, 'r') as f:
                for i, feature in enumerate(self.config.features):
                    feature_event_pulses = f[
                        self.config.transform + '/' + feature
                    ][idx]
                    batch_range = range(batch_size_dim_start, batch_size_dim_end)
                    pulses_range = range(len(feature_event_pulses))
                    for j, k in zip(batch_range, pulses_range):
                        X[j, 0:len(feature_event_pulses[k]), i] = feature_event_pulses[k]
                for i, target in enumerate(self.config.targets):
                    dataset_name = self.config.transform + '/' + target
                    if dataset_name in f:
                        event_targets = f[dataset_name][idx]
                    else:
                        event_targets = f['raw/' + target][idx]
                    for j, k in zip(batch_range, pulses_range):
                        y[j, i] = event_targets[k]
            batch_size_dim_start = batch_size_dim_end
        X = np.transpose(X, (0, 2, 1))
        return X, y
