from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import h5py as h5
import pandas as pd


class CnnGenerator(Sequence):
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
            file_name: list(self.ids[self.ids.file == file_name].idx.values)
            for file_name in self.ids.file.unique()
        }
        for i, (file, idx) in enumerate(ids_temp.to_numpy()):
            with h5.File(file, 'r') as f:
                event_indices = f['masks/' + self.config.mask][idx]
                event_length = len(event_indices)
                event_pulses = np.zeros(
                    (
                        event_length,
                        len(self.config.features)
                    )
                )
                for j, feature in enumerate(self.config.features):
                    event_pulses[:, j] = f[
                        self.config.transform + '/' + feature
                    ][idx][event_indices]
                X[i, 0:len(event_pulses), :] = event_pulses
                for k, target in enumerate(self.config.targets):
                    dataset_name = self.config.transform + '/' + target
                    if dataset_name in f:
                        y[i, k] = f[dataset_name][idx]
                    else:
                        y[i, k] = f['raw/' + target][idx]
        return X, y
