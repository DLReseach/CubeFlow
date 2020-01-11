from pathlib import Path
import h5py as h5
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.utils import get_project_root


class CnnSplit:
    def __init__(self, config):
        self.config = config
        if self.config.dev_run == True:
            self.config.no_of_files = 1
            self.config.batch_size = 2


    def get_files(self):
        root_path = get_project_root()
        data_path = root_path.joinpath('./data/' + self.config.data_type)
        data_files = [file for file in data_path.glob('*.h5')]
        if self.config.no_of_files != 0:
            data_files = data_files[0:self.config.no_of_files]
        return data_files


    def get_file_indices(self, file_list):
        output_dict = {str(file_name): [] for file_name in file_list}
        for file in file_list:
            with h5.File(file, 'r') as f:
                output_dict[str(file)].extend(
                    list(range(f['meta/events'][...]))
                )
        return output_dict


    def dom_trimmer(self, file_dict):
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
                            and doms >= self.config.min_doms
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


    def batcher(self, df):
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
        data_files = self.get_files()
        events_dict = self.get_file_indices(data_files)
        if self.config.max_doms is not None:
            events_dict = self.dom_trimmer(events_dict)
        events_df = self.pandas_index_list(events_dict)
        train_df, validate_df, test_df = self.splitter(events_df)
        if self.config.dev_run:
            train_df = train_df[0:self.config.no_dev_examples]
            validate_df = validate_df[0:self.config.no_dev_examples]
            test_df = test_df[0:self.config.no_dev_examples]
        train_list = self.batcher(train_df)
        validate_list = self.batcher(validate_df)
        test_list = self.batcher(test_df)
        return (train_df, validate_df, test_df), (train_list, validate_list, test_list)
