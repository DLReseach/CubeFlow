from pathlib import Path
import h5py as h5
import numpy as np
import pandas as pd
import itertools
import random
from sklearn.model_selection import train_test_split
from utils.utils import get_project_root
from utils.utils import get_time


class CnnPreprocess:
    def __init__(self, sets_list, config):
        self.sets_list = sets_list
        self.config = config
        if self.config.dev_run == True:
            self.config.no_of_files = 1
            self.config.batch_size = 2
    

    def file_trimmer(self):
        for file_set in self.sets_list:
            self.sets_list[file_set] = dict(
                itertools.islice(self.sets_list[file_set].items(),
                self.config.no_of_files
            )
        )
    

    def dev_examples_trimmer(self):
        for file_set in self.sets_list:
            for file in self.sets_list[file_set]:
                self.sets_list[file_set][file] = self.sets_list[file_set][file][0:self.config.no_dev_examples]


    def dom_trimmer(self):
        for dictionary in self.sets_list:
            for file in self.sets_list[dictionary]:
                idx = sorted(self.sets_list[dictionary][file])
                with h5.File(file, 'r') as f:
                    data = f['masks/' + self.config.mask][idx]
                    no_of_doms = [len(event) for event in data]
                    try:
                        cleaned_idx = [
                            idx[i] for i in range(len(idx))
                            if no_of_doms[i] <= self.config.max_doms
                            and no_of_doms[i] >= self.config.min_doms
                        ]
                        self.sets_list[dictionary][file] = cleaned_idx
                    except Exception:
                        print(
                            'Failed trimming doms.'
                            'Maybe you used too small a max value?'
                        )


    def batcher(self):
        batched_dict = {}
        for file_set in self.sets_list:
            batched_dict[file_set] = []
            for file_name in self.sets_list[file_set]:
                random.shuffle(self.sets_list[file_set][file_name])
                batches_in_file = int(
                    np.floor(
                        len(self.sets_list[file_set][file_name]) / self.config.batch_size
                    )
                )
                for batch in range(batches_in_file):
                    idx_start = self.config.batch_size * batch
                    idx_end = self.config.batch_size * (batch + 1)
                    grouped_batch = self.sets_list[file_set][file_name][idx_start:idx_end]
                    batched_dict[file_set].append({file_name: grouped_batch})
        return batched_dict


    def return_indices(self):
        if self.config.no_of_files > 0:
            print('Starting trimming files at', get_time())
            self.file_trimmer()
            print('Finished trimming files at', get_time())
        if self.config.max_doms is not None:
            print('Starting trimming DOMs at', get_time())
            self.dom_trimmer()
            print('Finished trimming DOMs at', get_time())
        if self.config.dev_run == True:
            print('Starting trimming dev examples at', get_time())
            self.dev_examples_trimmer()
            print('Finished trimming dev examples at', get_time())
        print('Starting batching files at', get_time())
        batched_sets_list = self.batcher()
        print('Finished batching files at', get_time())
        return batched_sets_list
