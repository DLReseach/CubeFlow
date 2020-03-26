import os
import torch
import importlib
from pathlib import Path
import socket

from src.modules.utils import create_experiment_name
from src.modules.utils import get_dirs_and_config
from src.modules.utils import get_time
from src.modules.mask_and_split import MaskAndSplit
from src.modules.reporter import Reporter
from src.modules.saver import Saver
from src.modules.trainer import Trainer

def main():
    experiment_name = create_experiment_name(slug_length=2)
    dirs, config = get_dirs_and_config(experiment_name, True)

    if socket.gethostname() == 'air.local':
        train_set = Path().home().joinpath('CubeFlowData').joinpath('dbs').joinpath('test_transformed.db')
        val_set = Path().home().joinpath('CubeFlowData').joinpath('dbs').joinpath('test_transformed.db')
        mask_and_split = MaskAndSplit(config, dirs, ['test'])
        sets = mask_and_split.split()
        sets['train'] = sets['test']
        sets['val'] = sets['test']
    elif socket.gethostname() == 'gpulab':
        train_set = Path('/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/train_transformed.db')
        val_set = Path('/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/val_transformed.db')
        mask_and_split = MaskAndSplit(config, dirs, ['train', 'val'])
        sets = mask_and_split.split()

    Loader = getattr(importlib.import_module('src.dataloaders.' + config['dataloader']), 'Dataloader')

    print(
        '{}: The overlap between train and val set is {}'
        .format(get_time(), len(list(set(sets['val']) & set(sets['train']))))
    )

    if 'SRTInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SRTInIcePulses'
        config['cleaning_length'] = 'srt_in_ice_pulses_event_length'
    elif 'SplitInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SplitInIcePulses'
        config['cleaning_length'] = 'split_in_ice_pulses_event_length'

    if config['dev_run']:
        sets['train'] = sets['train'][0:20000]
        sets['val'] = sets['val'][0:20000]

    train_dataset = Loader(
        sets['train'],
        config,
        train_set,
        test=False
    )
    val_dataset = Loader(
        sets['val'],
        config,
        val_set,
        test=True
    )
    reporter = Reporter(config, experiment_name)
    saver = Saver(config, dirs)

    Loss = getattr(importlib.import_module('src.losses.losses'), config['loss'])
    loss_init = Loss()
    loss = loss_init.loss
    Model = getattr(importlib.import_module('src.models.' + config['model']), 'Model')
    model = Model()
    Optimizer = getattr(importlib.import_module('src.optimizers.optimizers'), config['optimizer'])
    optimizer_init = Optimizer(model.parameters(), config['min_learning_rate'])
    optimizer = optimizer_init.optimizer

    trainer = Trainer(
        config,
        model,
        optimizer,
        loss,
        reporter,
        saver,
        train_dataset,
        val_dataset
    )

    trainer.fit()
    
    print('{}: Script done!'.format(get_time()))

if __name__ == '__main__':
    main()
