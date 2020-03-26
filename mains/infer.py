import os
import torch
import importlib
from pathlib import Path
import argparse
from torchsummary import summary

from src.modules.utils import get_dirs_and_config
from src.modules.utils import get_time
from src.modules.mask_and_split import MaskAndSplit
from src.modules.inferer import Inferer
from src.modules.transform_inverter import DomChargeScaler
from src.modules.transform_inverter import EnergyNoLogTransformer
from src.modules.truth_saver import TruthSaver
from src.modules.error_calculator import ErrorCalculator
from src.modules.histogram_calculator import HistogramCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', help='run name')
    args = parser.parse_args()
    experiment_name = args.run

    test_set_transformed_path = Path().home().joinpath('CubeFlowData').joinpath('dbs').joinpath('test_transformed.db')

    dirs, config = get_dirs_and_config(experiment_name, False)

    errors_db_path = dirs['dbs'].joinpath('errors.db')
    predictions_db_path = dirs['dbs'].joinpath('predictions.db')

    mask_and_split = MaskAndSplit(config, dirs, ['test'])
    sets = mask_and_split.split()

    config['val_batch_size'] = 2000

    Loader = getattr(importlib.import_module('src.dataloaders.' + config['dataloader']), 'Dataloader')

    if 'SRTInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SRTInIcePulses'
        config['cleaning_length'] = 'srt_in_ice_pulses_event_length'
    elif 'SplitInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SplitInIcePulses'
        config['cleaning_length'] = 'split_in_ice_pulses_event_length'

    sets['test'] = sets['test'][0:20000]

    dataset = Loader(
        sets['test'],
        config,
        test_set_transformed_path,
        test=True
    )

    events = [item for sublist in dataset.events for item in sublist]

    if not dirs['dbs'].joinpath('predictions.db').is_file():
        print('{}: First run with these masks; saving truth and retro_crs_prefit to prediction db'.format(get_time()))
        TruthSaver(config, dirs, events)

    Loss = getattr(importlib.import_module('src.losses.losses'), config['loss'])
    loss_init = Loss()
    loss = loss_init.loss
    Model = getattr(importlib.import_module('src.models.' + config['model']), 'Model')
    model = Model()
    Optimizer = getattr(importlib.import_module('src.optimizers.optimizers'), config['optimizer'])
    optimizer_init = Optimizer(model.parameters(), config['min_learning_rate'])
    optimizer = optimizer_init.optimizer

    inferer = Inferer(model, optimizer, loss, dataset, config, experiment_name, dirs)
    model_path = dirs['run'].joinpath('model.pt')

    print('{}: Beginning inference'.format(get_time()))
    inferer.infer(model_path, dirs['run'])

    print('{}: Beginning error calculation'.format(get_time()))
    if not dirs['dbs'].joinpath('errors.db').is_file():
        print('{}: First run with these masks; calculating retro_crs_prefit errors'.format(get_time()))
        ErrorCalculator('retro_crs_prefit', dirs)
    ErrorCalculator(experiment_name, dirs)

    print('{}: Beginning histogram calculation'.format(get_time()))
    HistogramCalculator(experiment_name, dirs)

    print('{}: Script done!'.format(get_time()))

if __name__ == '__main__':
    main()
