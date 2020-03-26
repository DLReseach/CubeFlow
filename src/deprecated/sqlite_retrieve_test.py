from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3
import pickle
import torch
from bunch import Bunch

from src.modules.sql_dataloader import SqlDataset
from src.modules.pickle_dataloader import PickleDataset


def write_to_csv(X, events, features):
    '''Write coerced batches to csv file for testing.

    Args:
        X (numpy.ndarray): Coerced array containing batched padded features.
        z (numpy.ndarray): Event numbers and rows used for testing.
        features (list): List of fetched features.
    
    Returns:
        None
    '''
    test_df = pd.DataFrame(columns=features)
    X = X.numpy()
    X = X.reshape(-1, len(features))
    df = pd.DataFrame(
        {key: X[:, value] for value, key in enumerate(features)}
    )
    df.to_csv(str(Path().home().joinpath('csv_test.csv')), index=False)


# Choose what features to fetch
features = [
    'dom_x',
    'dom_y',
    'dom_z',
    'dom_time',
    'dom_charge'
]
# Choose what targets to fetch
targets = [
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
    'true_primary_energy',
    'true_primary_time',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z'
]
comparisons = [
    'retro_crs_prefit_azimuth',
    'retro_crs_prefit_zenith',
    'retro_crs_prefit_energy',
    'retro_crs_prefit_time',
    'retro_crs_prefit_x',
    'retro_crs_prefit_y',
    'retro_crs_prefit_z'
]
# Choose cleaning method; SRTInIcePulses / SplitInIcePulses
cleaning = 'SRTInIcePulses'

# Path where database lives
sql_path = Path('/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/')
sql_file = sql_path.joinpath('train_set.db')

# Number of events in database
events = list(range(0, 9021092))

# Open and read in relevant masks
masks_path = Path('/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/masks')
particle_mask_file = masks_path.joinpath('muon_neutrino.pickle')
length_mask_file = masks_path.joinpath('dom_interval_SRTInIcePulses_min0_max200.pickle')
with open(str(particle_mask_file), 'rb') as f:
    particle_mask = pickle.load(f)
with open(str(length_mask_file), 'rb') as f:
    length_mask = pickle.load(f)

# Get intersection of events in masks
mask = list(set(events) & set(particle_mask) & set(length_mask))

# Choose whether to write out one batch to a csv file
write_out_to_file = False

# Set parameters
max_doms = 200
batch_size = 64
reporting_interval = 1000
total_batches = len(mask) // batch_size
# Choose when fetching stops
no_of_batches = 10000

total_events_per_second = []

start_script = datetime.now()

workers = 4

config = Bunch(
    batch_size=batch_size,
    gpulab_data_dir=Path('/home/bjoernhm/CubeML/data'),
    data_type='oscnext-genie-level5-v01-01-pass2',
    mask=cleaning,
    features=features,
    targets=targets,
    comparison_metrics=comparisons
)

print('{} workers'.format(workers))
# Setup training set
training_set = PickleDataset(mask, config)
# Create dataloader; batch_size and shuffle are None because we batch and shuffle ourselves
training_generator = torch.utils.data.DataLoader(training_set, batch_size=None, shuffle=False, num_workers=workers)
# Total number of events
no_of_events = batch_size * no_of_batches
events_per_second = []
# Loop until no_of_batches

for i, batch in enumerate(training_generator):
    if i % reporting_interval == 0:
        if i == 0:
            start_minibatch = datetime.now()
        else:
            end_minibatch = datetime.now()
            delta_time_minibatch = round((end_minibatch - start_minibatch).total_seconds(), 2)
            events_per_second.append(batch_size * reporting_interval // delta_time_minibatch)
            start_minibatch = datetime.now()
            print('Fetched batch', i)
    fetched = batch
    # if i % no_of_batches == 0 and i > 0:
    #     break
total_events_per_second.append(events_per_second)

print(
    'Workers: {} events, {} events per second'
    .format(
        no_of_batches * batch_size,
        sum(events_per_second) // len(events_per_second)
    )
)

end_script = datetime.now()
delta_time_script = round((end_script - start_script).total_seconds(), 2)

print(
    'Script done; {} events per second on average; took {} seconds'
    .format(
        sum(events_per_second) // len(events_per_second),
        delta_time_script
    )
)

# If testing, write out last batch to csv file
if write_out_to_file:
    write_to_csv(X, events, features)
