import numpy as np
import keras
import h5py as h5

class CnnGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,
        ids,
        features,
        targets,
        max_doms,
        mask,
        transform,
        batch_size,
        n_channels=5,
        shuffle=True
    ):
        'Initialization'
        self.ids = ids
        self.features = features
        self.targets = targets
        self.max_doms = max_doms
        self.mask = mask
        self.transform = transform
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indices[
            index * self.batch_size:(index + 1) * self.batch_size
            ]
        # Find list of IDs
        ids_temp = [self.ids[k] for k in indices]
        # Generate data
        X, y = self.__data_generation(ids_temp)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indices = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


    def __data_generation(self, ids_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.max_doms, self.n_channels))
        y = np.empty((self.batch_size, len(self.features)))
        # Generate data
        for i, idx in enumerate(ids_temp):
            with h5.File(ids_temp.file, 'r') as f:
                event_indices = f['masks/' + mask][idx]
                event_length = len(event_indices)
                event_pulses = np.zeros((event_length, len(features)))
                for j, feature in enumerate(features):
                    event_pulses[:, j] = f[
                        transform + '/' + feature
                    ][idx][event_indices]
                X[i,] = event_pulses
                target_values = np.zeros(len(self.features))
                for k, target in enumerate(targets):
                    target_values[k] = f[transform + '/' + target]
                y[i,] = target_values
        return X, y
