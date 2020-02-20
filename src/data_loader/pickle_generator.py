import torch
import numpy as np
import pickle
from pathlib import Path
from operator import itemgetter
from utils.utils import get_project_root


class PickleGenerator(torch.utils.data.Dataset):
    def __init__(self, config, ids, test, conv_type):
        self.config = config
        self.ids = ids
        self.test = test
        self.conv_type = conv_type
        if self.config.gpulab:
            self.data_dir = Path(self.config.gpulab_data_dir).joinpath(
                self.config.data_type + '/pickles'
            )
        else:
            self.data_dir = Path.home().joinpath(
                self.config.data_dir +
                self.config.data_type +
                '/pickles'
            )
        self.file_extension = '.pickle'
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.ids))

    def __getitem__(self, index):
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
        if self.test:
            comparisons = {}
            for i, comparison_type in enumerate(self.config.comparison_metrics):
                comparison = self.config.opponent + '_' + comparison_type
                comparisons[comparison_type] = torch.tensor(loaded_file['raw'][comparison])
            energy = loaded_file['raw']['true_primary_energy']
        if self.conv_type == 'conv1d':
            X = np.transpose(X, (1, 0))
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        if self.test:
            return X, y, comparisons, energy, event_length
        else:
            return X, y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.ids))
        if self.config.shuffle and self.test:
            np.random.shuffle(self.indices)

    def check_entry_in_transform(self, dictionary, entry, comparison):
        if entry in dictionary[comparison]:
            transform = comparison
        elif entry in dictionary['raw']:
            transform = 'raw'
        else:
            print('Whoops! The feature ain\'t in the file!')
        return transform

