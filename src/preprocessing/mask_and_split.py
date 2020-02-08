import pickle


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
    
    def get_intersection(self):
        masks_dict_keys = list(self.masks_dict.keys())
        self.intersection = list(
            set(self.masks_dict[masks_dict_keys[0]])
            & set(self.masks_dict[masks_dict_keys[1]])
        )
        if len(masks_dict_keys) > 2:
            for i in range(1, len(masks_dict_keys) - 1):
                self.intersection = list(
                    set(self.intersection)
                    & set(self.masks_dict[masks_dict_keys[i + 1]])
                )

    def split(self):
        fraction_sum = self.config.train_fraction + self.config.val_fraction + self.config.test_fraction
        assert fraction_sum <= 1.0, "Oh no! Split fraction sum is greater than 1!"
        sets = {}
        no_of_events = len(self.intersection)
        train_share = int(self.config.train_fraction * no_of_events)
        val_share = int(self.config.val_fraction * no_of_events)
        test_share = int(self.config.test_fraction * no_of_events)
        sets['train'] = self.intersection[:train_share]
        sets['val'] = self.intersection[train_share:(train_share + val_share)]
        sets['test'] = self.intersection[(train_share + val_share):(train_share + val_share + test_share)]
        return sets
