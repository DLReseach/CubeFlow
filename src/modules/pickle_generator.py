import torch
import numpy as np
import pickle
import shelve
from pathlib import Path
from operator import itemgetter

from src.modules.utils import get_project_root


class PickleGenerator(torch.utils.data.Dataset):
    def __init__(self, config, ids, test, val, conv_type):
        self.config = config
        self.ids = ids
        self.test = test
        self.val = val
        self.conv_type = conv_type
        if self.config.gpulab:
            if self.config.file_type == 'pickle':
                self.data_dir = Path(self.config.gpulab_data_dir).joinpath(
                    self.config.data_type + '/pickles'
                )
                self.file_extension = '.pickle'
            elif self.config.file_type == 'shelve':
                self.data_dir = Path(self.config.gpulab_data_dir).joinpath(
                    self.config.data_type + '/shelve/cube_shelve'
                )
                self.db = shelve.open(str(self.data_dir), 'r')
        else:
            if self.config.file_type == 'pickle':
                self.data_dir = Path.home().joinpath(
                    self.config.data_dir +
                    self.config.data_type +
                    '/pickles'
                )
                self.file_extension = '.pickle'
            elif self.config.file_type == 'shelve':
                self.data_dir = Path.home().joinpath(
                    self.config.data_dir +
                    self.config.data_type +
                    '/shelve/cube_shelve'
                )
                self.db = shelve.open(str(self.data_dir), 'r')

        self.on_epoch_end()

    def __len__(self):
        return int(len(self.ids))

    def __getitem__(self, index):
        if self.config.file_type == 'pickle':
            output = self.retrieve_from_pickle(index)
        elif self.config.file_type == 'shelve':
            output = self.retrieve_from_shelve(index)
        return output

    def retrieve_from_pickle(self, index):
        max_doms = self.config.max_doms
        no_features = len(self.config.features)
        no_targets = len(self.config.targets)
        file_number = str(self.ids[self.indices[index]])
        sub_folder = str(int(np.floor(int(file_number) / 10000) % 9999))
        X = np.zeros((max_doms, no_features))
        y = np.zeros((no_targets))
        comparisons = np.zeros((len(self.config.comparison_metrics)))
        file = self.data_dir.joinpath(
            sub_folder + '/' + file_number + self.file_extension
        )
        with open(file, 'rb') as f:
            loaded_file = pickle.load(f)
        event_mask = loaded_file['masks'][self.config.mask]
        event_length = len(event_mask)
        for i, feature in enumerate(self.config.features):
            transform = self.check_entry_in_transform(
                loaded_file,
                feature,
                self.config.transform
            )
            X[0:event_length, i] = loaded_file[transform][feature][event_mask]
        for i, target in enumerate(self.config.targets):
            transform = self.check_entry_in_transform(
                loaded_file,
                target,
                self.config.transform
            )
            y[i] = loaded_file[transform][target]
        if self.test or self.val:
            comparisons = []
            for i, comparison_type in enumerate(self.config.comparison_metrics):
                comparison = self.config.opponent + '_' + comparison_type
                comparisons.append(loaded_file['raw'][comparison])
        if self.test or self.val or self.config.save_train_dists:
            energy = loaded_file['raw']['true_primary_energy']
        if self.conv_type == 'conv1d':
            X = np.transpose(X, (1, 0))
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        if self.test or self.val:
            return X, y, comparisons, energy, event_length, file_number
        elif not (self.test or self.val) and self.config.save_train_dists:
            return X, y, energy, event_length
        else:
            return X, y, [], []

    def retrieve_from_shelve(self, index):
        max_doms = self.config.max_doms
        no_features = len(self.config.features)
        no_targets = len(self.config.targets)
        event_number = str(self.ids[self.indices[index]])
        X = np.zeros((max_doms, no_features))
        y = np.zeros((no_targets))
        comparisons = np.zeros((len(self.config.comparison_metrics)))
        loaded_event = self.db[event_number]
        event_mask = loaded_event['masks'][self.config.mask]
        event_length = len(event_mask)
        for i, feature in enumerate(self.config.features):
            transform = self.check_entry_in_transform(
                loaded_event,
                feature,
                self.config.transform
            )
            X[0:event_length, i] = loaded_event[transform][feature][event_mask]
        for i, target in enumerate(self.config.targets):
            transform = self.check_entry_in_transform(
                loaded_event,
                target,
                self.config.transform
            )
            y[i] = loaded_event[transform][target]
        if self.test:
            comparisons = []
            for i, comparison_type in enumerate(self.config.comparison_metrics):
                comparison = self.config.opponent + '_' + comparison_type
                comparisons.append(loaded_event['raw'][comparison])
        if self.test or self.config.save_train_dists:
            energy = loaded_event['raw']['true_primary_energy']
        if self.conv_type == 'conv1d':
            X = np.transpose(X, (1, 0))
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        if self.test:
            return X, y, comparisons, energy, event_length, event_number
        elif not self.test and self.config.save_train_dists:
            return X, y, energy, event_length
        else:
            return X, y, [], []

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        if self.config.shuffle and self.test:
            np.random.shuffle(self.indices)

    def close_db(self):
        self.db.close()

    def check_entry_in_transform(self, dictionary, entry, comparison):
        if entry in dictionary[comparison]:
            transform = comparison
        elif entry in dictionary['raw']:
            transform = 'raw'
        else:
            print('Whoops! The feature ain\'t in the file!')
        return transform

