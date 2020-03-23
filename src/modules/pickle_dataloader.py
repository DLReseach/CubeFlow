import torch
import sqlite3
import numpy as np
import pickle
from pathlib import Path


class PickleDataset(torch.utils.data.Dataset):
    '''Dataset fetching from SQLite database.'''
    def __init__(self, mask, config, set_type):
        self.mask = mask
        self.config = config
        if set_type == 'train':
            self.batch_size = config.batch_size
        elif set_type == 'val' or set_type == 'test':
            self.batch_size = config.val_batch_size
        self.floor_events = int(len(mask) // self.batch_size * self.batch_size)
        self.no_of_batches = int(len(mask) // self.batch_size)
        self.on_epoch_start()

    def __len__(self):
        '''Standard __len__.'''
        return self.no_of_batches

    def on_epoch_start(self):
        '''Shuffle events at epoch start.'''
        # Shuffle events list
        np.random.shuffle(self.mask)
        self.events = []
        # Batch events in batch sizes
        for i in range(0, self.floor_events, self.batch_size):
            self.events.append(self.mask[i:i + self.batch_size])

    def __getitem__(self, index):
        '''Standard __getitem__.
        
        Args:
                index (int): Batch index.

        Returns:
                tuple: tuple containing:
                    X (numpy.ndarray): Coerced array containing batched padded features.
                    y (numpy.ndarray): Batched targets.
                    z (numpy.ndarray): Event numbers and rows used for testing.
        '''
        # Retrieve batch from events list
        batch_events = self.events[index]
        X, y, events = self._coerce_batch(batch_events)
        print(X)
        return X, y, events

    def _get_from_pickle(self, events):
        sequentials = []
        scalars = []
        lengths = []
        masks = []
        for event in events:
            sub_folder = str(int(event // 10000 % 9999))
            file = Path(self.config.gpulab_data_dir).joinpath(self.config.data_type).joinpath('pickles').joinpath(sub_folder).joinpath(str(event) + '.pickle')
            with open(file, 'rb') as f:
                loaded_file = pickle.load(f)
            event_mask = loaded_file['masks'][self.config.mask]
            masks.append(event_mask)
            event_length = len(event_mask)
            lengths.append(event_length)
            event_sequential = np.zeros((event_length, len(self.config.features)))
            event_scalar = np.zeros((1, len(self.config.targets)))
            for i, feature in enumerate(self.config.features):
                try:
                    event_sequential[:, i] = loaded_file[self.config.transform][feature][event_mask]
                except:
                    event_sequential[:, i] = loaded_file['raw'][feature][event_mask]
            for i, target in enumerate(self.config.targets):
                try:
                    event_scalar[:, i] = loaded_file[self.config.transform][target]
                except:
                    event_scalar[:, i] = loaded_file['raw'][target]
            sequentials.append(event_sequential)
            scalars.append(event_scalar)
        return sequentials, scalars, lengths, masks

    def _coerce_batch(self, events):
        '''Retrieve events from a SQLite database and coerce + pad them.

        Args:
                events (list): List of events to retrieve.
        
        Returns:
                tuple: tuple containing:
                    X (numpy.ndarray): Coerced array containing batched padded features.
                    y (numpy.ndarray): Batched targets.
                    z (numpy.ndarray): Event numbers and rows used for testing.
        '''
        # Get the rows from the database
        sequentials, scalars, lengths, masks = self._get_from_pickle(events)
        # Length of longest event, used for padding
        # max_length = max(lengths)
        max_length = 200
        # Preallocation of arrays
        X = np.zeros((len(events), max_length, len(self.config.features)))
        y = np.zeros((len(events), len(self.config.targets)))
        for i, (sequential, scalar) in enumerate(zip(sequentials, scalars)):
            event_length = lengths[i]
            insert_point = int((max_length - event_length) // 2)
            X[i, insert_point:insert_point + event_length, :] = sequential
            y[i, :] = scalar
        X = np.transpose(X, axes=[0, 2, 1])
        events = np.array(events).reshape(self.batch_size, -1)
        return X, y, events
