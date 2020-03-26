import pickle
import numpy as np


class MaskAndSplit:
    def __init__(self, config, files_and_dirs, set_types):
        super().__init__()
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.sets = set_types
        self.get_masks()

    def get_masks(self):
        self.masks_dict = {}
        for mask in self.config['masks']:
            mask_file = self.files_and_dirs['masks'].joinpath(mask + '.pickle')
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
        sets = {}
        for set_type in self.sets:
            set_file = self.files_and_dirs['masks'].joinpath(set_type + '_set.pkl')
            with open(set_file, 'rb') as f:
                sets[set_type] = pickle.load(f)
            sets[set_type] = self.get_intersection(sets[set_type])
        return sets
