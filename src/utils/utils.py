import argparse
from pathlib import Path
import time
import datetime
from coolname import generate_slug
import numpy as np
import h5py as h5


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def get_time():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return st


def print_data_set_sizes(
    config,
    train_generator,
    validation_generator,
    test_generator
):
    print(
        'We have around {} training events'.format(
            len(train_generator) * config.batch_size
        )
    )
    print(
        'We have around {} validation events'.format(
            len(validation_generator) * config.batch_size
        )
    )
    print(
        'We have around {} test events'.format(
            len(test_generator) * config.batch_size
        )
    )


def create_experiment_name(config, slug_length):
    cool_name = generate_slug(2)
    today = str(datetime.date.today())
    experiment_name = config.exp_name + '_' + today + '.' + cool_name
    return experiment_name


def set_random_seed():
    np.random.seed(int(time.time()))


def h5_groups_reader(data_file, group):
    with h5.File(data_file, 'r') as f:
        groups = list(f[group].keys())
    return groups


def h5_data_reader(data_file, group, idx):
    with h5.File(data_file, 'r') as f:
        if idx == 'all':
            data = f[group][:]
        else:
            data = f[group][idx]
    return data
