import pickle
import numpy as np


class MaskAndSplit:
    def __init__(self, config, files_and_dirs):
        super().__init__()
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.get_masks()

    def get_masks(self):
        self.masks_dict = {}
        for mask in self.config.masks:
            mask_file = self.files_and_dirs['masks_dir'].joinpath(mask + '.pickle')
            with open(mask_file, 'rb') as f:
                self.masks_dict[mask] = pickle.load(f)
    
    def get_intersection(self, events_set):
        for key in self.masks_dict.keys():                
            events_set = list(
                set(events_set)
                & set(self.masks_dict[key])
            )
        return events_set

    def split(self):
        n_events = 11308407
        events = np.arange(n_events)
        train_indices_max = int(np.floor(0.8 * n_events))
        val_indices_max = int(n_events - (n_events - train_indices_max) * 0.5)
        train_indices = events[0:train_indices_max]
        val_indices = events[train_indices_max:val_indices_max]
        test_indices = events[val_indices_max:]
        sets = {}
        sets['train'] = self.get_intersection(train_indices)
        sets['val'] = self.get_intersection(val_indices)
        sets['test'] = self.get_intersection(test_indices)
        if self.config.dev_run:
            sets['train'] = sets['train'][0:int(0.01 * len(sets['train']))]
            sets['val'] = sets['val'][0:int(0.01 * len(sets['val']))]
            sets['test'] = sets['test'][0:int(0.01 * len(sets['test']))]
        return sets
