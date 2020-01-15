from pathlib import Path
import h5py as h5
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.utils import get_project_root


class CnnSplit:
    def __init__(self, sets_list, config):
        self.sets_list = sets_list
        self.config = config
        if self.config.dev_run == True:
            self.config.no_of_files = 1
            self.config.batch_size = 2


    def dom_trimmer(self):
        for dictionary in self.sets_list:
            for file in dictionary:
                idx = sorted(dictionary[file])
                with h5.File(file, 'r') as f:
                    no_of_doms = [
                        len(event) for event in f['masks/' + self.config.mask][idx]
                    ] 
                    try:
                        length, indices = zip(
                            *(
                                (doms, index) for doms, index
                                in zip(no_of_doms, dictionary[file])
                                if doms <= self.config.max_doms
                                and doms >= self.config.min_doms
                            )
                        )
                        dictionary[file] = list(indices)
                    except Exception:
                        print(
                            'Failed trimming doms.'
                            'Maybe you used too small a max value?'
                        )
        return output_dict


    def batcher(self):
        set_list = []
        for file_name in df.file.unique():
            events_in_file = df[df['file'] == file_name]
            batches_in_file = int(
                np.floor(
                    len(events_in_file) / self.config.batch_size
                )
            )
            for batch in range(batches_in_file):
                idx_start = self.config.batch_size * batch
                idx_end = self.config.batch_size * (batch + 1)
                grouped_batch = events_in_file.idx.iloc[idx_start:idx_end].values.tolist()
                set_list.append({file_name: grouped_batch})
        return set_list


    def return_indices(self):
        if self.config.max_doms is not None:
            sets_list = self.dom_trimmer(self.sets_list)
        if self.config.dev_run:
            train_df = train_df[0:self.config.no_dev_examples]
            validate_df = validate_df[0:self.config.no_dev_examples]
            test_df = test_df[0:self.config.no_dev_examples]
        train_list = self.batcher(train_df)
        validate_list = self.batcher(validate_df)
        test_list = self.batcher(test_df)
        return (train_df, validate_df, test_df), (train_list, validate_list, test_list)
