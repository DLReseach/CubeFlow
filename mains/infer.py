import os
import torch
import importlib
from pathlib import Path
import argparse

from src.modules.utils import get_dirs_and_config
from src.modules.utils import get_time
from src.modules.mask_and_split import MaskAndSplit
from src.modules.inferer import Inferer
from src.modules.transform_inverter import DomChargeScaler
from src.modules.transform_inverter import EnergyNoLogTransformer
from src.modules.truth_saver import TruthSaver
from src.modules.error_calculator import ErrorCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', help='run name')
    args = parser.parse_args()
    experiment_name = args.run

    test_set_transformed_path = Path().home().joinpath('CubeFlowData').joinpath('dbs').joinpath('test_transformed.db')

    dirs, config, first_run = get_dirs_and_config(experiment_name, False)

    errors_db_path = dirs['dbs'].joinpath('errors.db')
    predictions_db_path = dirs['dbs'].joinpath('predictions.db')

    mask_and_split = MaskAndSplit(config, dirs, ['test'])
    sets = mask_and_split.split()

    config['val_batch_size'] = 2000

    if 'dataloader' not in config:
        Dl = getattr(importlib.import_module('src.modules.' + 'sql_dataloader_middle_pad'), 'Dataloader')
    else:
        Dl = getattr(importlib.import_module('src.modules.' + config['dataloader']), 'Dataloader')

    if 'SRTInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SRTInIcePulses'
        config['cleaning_length'] = 'srt_in_ice_pulses_event_length'
    elif 'SplitInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SplitInIcePulses'
        config['cleaning_length'] = 'split_in_ice_pulses_event_length'

    # sets['test'] = sets['test'][0:20000]

    dataset = Dl(
        sets['test'],
        config,
        test_set_transformed_path,
        test=True
    )

    events = [item for sublist in dataset.events for item in sublist]

    if first_run:
        print('{}: First run with these masks; saving truth and retro_crs_prefit to prediction db'.format(get_time()))
        TruthSaver(config, dirs, events)

    loss = torch.nn.MSELoss()
    Model = getattr(importlib.import_module('src.modules.' + config['model']), 'Model')
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['min_learning_rate'])

    inferer = Inferer(model, optimizer, loss, dataset, config, experiment_name, dirs)
    model_path = dirs['run'].joinpath('model.pt')

    print('{}: Beginning inference'.format(get_time()))
    inferer.infer(model_path, dirs['run'])

    print('{}: Beginning error calculation'.format(get_time()))
    ErrorCalculator(experiment_name, first_run, dirs)

    print('{}: Script done!'.format(get_time()))

if __name__ == '__main__':
    main()
