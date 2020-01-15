from pathlib import Path
import h5py as h5
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from utils.utils import get_project_root


class CnnSplit:
    def __init__(
        self,
        data_type,
        particle_type,
        test_fraction,
        validation_fraction,
        random_state
    ):
        self.data_type = data_type
        self.particle_type = particle_type
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.random_state = random_state


    def get_files(self):
        root_path = get_project_root()
        data_path = root_path.joinpath('./data/' + self.data_type)
        data_files = [
            file for file in data_path.glob('*' + self.particle_type + '*.h5')
        ]
        return data_files


    def get_file_indices(self, file_list):
        output_dict = {str(file_name): [] for file_name in file_list}
        for file in file_list:
            with h5.File(file, 'r') as f:
                output_dict[str(file)].extend(
                    list(range(f['meta/events'][...]))
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
            test_size=self.test_fraction,
            random_state=self.random_state
        )
        train, validate = train_test_split(
            train,
            test_size=self.validation_fraction,
            random_state=self.random_state
        )
        output_dict = {}
        output_dict['train'] = train
        output_dict['validate'] = validate
        output_dict['test'] = test
        return output_dict


    def dict_creator(self, dfs_dict):
        output_dict = {}
        for df in dfs_dict:
            set_dict = {}
            for file in dfs_dict[df].file.unique():
                set_dict[file] = dfs_dict[df][dfs_dict[df].file == file].idx.values
            output_dict[df] = set_dict
        return output_dict


    def save_sets(self):
        data_files = self.get_files()
        events_dict = self.get_file_indices(data_files)
        events_df = self.pandas_index_list(events_dict)
        dfs_dict = self.splitter(events_df)
        sets_list = self.dict_creator(dfs_dict)
        out_file = get_project_root().joinpath(
            'sets/' + self.particle_type + '.joblib'
        )
        joblib.dump(sets_list, out_file)


print('Starting set splitting...')
splitting = CnnSplit(
    'oscnext-genie-level5-v01-01-pass2',
    '120000',
    0.2,
    0.2,
    29897070
)
splitting.save_sets()
print('Set splitting done.')
