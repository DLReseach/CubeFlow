import h5py as h5
import joblib
import numpy as np

from utils.utils import get_project_root
from utils.utils import h5_groups_reader
from utils.utils import h5_data_reader

root = get_project_root()
sets_dir = root.joinpath('sets')
sets_files = [f for f in sets_dir.glob('**/*.joblib')]
transforms = ['raw', 'transform1']
current_file = joblib.load(sets_files[0])


def data_collector(set_file, transforms, set_type):
    data_dict = {}
    files_and_idx = set_file[set_type]
    for transform in transforms:
        data_dict[transform] = {}
        groups = h5_groups_reader(
            list(files_and_idx.keys())[0],
            transform
        )
        for group in groups:
            data_dict[transform][group] = []
        for i, file in enumerate(files_and_idx):
            print(len(files_and_idx) - i)
            with h5.File(file, 'r') as f:
                idx = sorted(files_and_idx[file])
                for group in groups:
                    data = f[transform + '/' + group][idx]
                    if type(data[0]) == np.ndarray:
                        data = np.concatenate(data).ravel()
                        data_dict[transform][group].extend(data)
                    else:
                        data_dict[transform][group].extend(data)
        print(data_dict[transforms[0]][groups[0]])
        break


data_collector(current_file, transforms, 'train')