from pathlib import Path
import h5py as h5
import numpy as np
import pandas as pd
from utils.utils import get_project_root
from sklearn.model_selection import train_test_split


class CnnSplit:
    def __init__(self, config):
        self.config = config
        if self.config.dev_run == True:
            self.config.no_of_files = 1
        self.return_indices()


    def get_files(self):
        root_path = get_project_root()
        data_path = root_path.joinpath('./data/' + self.config.data_type)
        data_files = [file for file in data_path.glob('*.h5')]
        if self.config.no_of_files != 0:
            data_files = data_files[0:self.config.no_of_files]
        return data_files


    def get_no_events(self, file_list, *datasets):
        output_dict = {key: 0 for key in datasets}
        for dataset in datasets:
            for file in file_list:
                with h5.File(file, 'r') as f:
                    no_of_events_in_file = f[dataset][...].item()
                    output_dict[dataset] += no_of_events_in_file
        return output_dict


    def get_file_indices(self, file_list):
        output_dict = {str(file_name): [] for file_name in file_list}
        for file in file_list:
            with h5.File(file, 'r') as f:
                output_dict[str(file)].extend(
                    list(range(f['meta/events'][...]))
                )
        return output_dict


    def max_dom_trimmer(self, file_dict):
        output_dict = {}
        for file in file_dict:
            with h5.File(file, 'r') as f:
                no_of_doms = [
                    len(event) for event in f['masks/' + self.config.mask][:]
                ]
                try:
                    length, indices = zip(
                        *(
                            (doms, index) for doms, index
                            in zip(no_of_doms, file_dict[file])
                            if doms <= self.config.max_doms
                        )
                    )
                    output_dict[str(file)] = list(indices)
                except Exception:
                    print(
                        'Failed trimming doms.'
                        'Maybe you used too small a value?'
                    )
        return output_dict


    def pandas_index_list(self, file_dict):
        indices_df = pd.DataFrame(
            [
                (file, index) for (file, L) in file_dict.items()
                for index in L
            ],
            columns=['file', 'idx']
        )
        return indices_df


    def splitter(self, file_df):
        train, test = train_test_split(
            file_df,
            test_size=self.config.test_fraction,
            random_state=self.config.random_state
        )
        train, validate = train_test_split(
            train,
            test_size=self.config.validation_fraction,
            random_state=self.config.random_state
        )
        return train, validate, test


    def return_indices(self):
        data_files = self.get_files()
        events_dict = self.get_file_indices(data_files)
        if self.config.max_doms is not None:
            events_dict = self.max_dom_trimmer(events_dict)
        events_df = self.pandas_index_list(events_dict)
        train, validate, test = self.splitter(events_df)
        if self.config.dev_run:
            train = train[0:self.config.no_dev_examples]
            validate = validate[0:self.config.no_dev_examples]
            test = test[0:self.config.no_dev_examples]
        return train, validate, test
