import pickle
import random
import numpy as np


class MaskAndSplit:
    def __init__(self, config, files_and_dirs):
        super().__init__()
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.get_masks()
        self.get_intersection()

    def get_masks(self):
        self.masks_dict = {}
        for mask in self.config.masks:
            mask_file = self.files_and_dirs['masks_dir'].joinpath(mask + '.pickle')
            with open(mask_file, 'rb') as f:
                self.masks_dict[mask] = pickle.load(f)
    
    def get_intersection(self, events_set):
        masks_dict_keys = list(self.masks_dict.keys())
        for key in masks_dict.keys():                
            events_set = list(
                set(events_set)
                & set(self.masks_dict[key])
            )
        return events_set

    def split(self):
        n_events = 11308407
        train_indices_max = int(np.floor(0.8 * n_events))
        val_indices_max = int(n_events - (n_events - train_indices_max) * 0.5)
        indices_shuffled = np.arange(n_events)
        seed = 2912
        random.seed(seed)
        random.shuffle(indices_shuffled)
        train_indices = indices_shuffled[0:train_indices_max]
        val_indices = indices_shuffled[train_indices_max:val_indices_max]
        test_indices = indices_shuffled[val_indices_max:]
        print(len(train_indices))
        print(len(val_indices))
        print(len(test_indices))
        sets = {}
        sets['train'] = self.get_intersection(train_indices)
        sets['val'] = self.get_intersection(val_indices)
        sets['test'] = self.get_intersection(test_indices)
        print(len(sets['train']))
        print(len(sets['val']))
        print(len(sets['test']))
        return sets
