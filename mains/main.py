import os
import torch
# from torch.utils.data import DataLoader
# from pytorch_lightning import Trainer
# from pytorch_lightning.callbacks import EarlyStopping
import wandb as wandb
import matplotlib.cbook
import warnings
from argparse import Namespace
import slack
import numpy as np
import importlib
# from torch_lr_finder import LRFinder

# from src.modules.cnn_conv1d import CnnSystemConv1d
from src.modules.config import process_config
from src.modules.utils import get_args
from src.modules.utils import create_experiment_name
from src.modules.utils import get_files_and_dirs
from src.modules.utils import get_time
from src.modules.mask_and_split import MaskAndSplit
from src.modules.reporter import Reporter
from src.modules.saver import Saver
from src.modules.resolution_comparison import ResolutionComparison
from src.modules.trainer import Trainer

def main():
    experiment_name = create_experiment_name(slug_length=2)
    dirs, config, _ = get_dirs_and_config(experiment_name)

    train_set = Path('/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/train_transformed.db')
    val_set = Path('/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/val_transformed.db')

    mask_and_split = MaskAndSplit(config, dirs, ['train', 'val'])
    sets = mask_and_split.split()

    Dl = getattr(importlib.import_module('src.modules.' + config['dataloader']), 'Dataloader')

    if 'SRTInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SRTInIcePulses'
        config['cleaning_length'] = 'srt_in_ice_pulses_event_length'
    elif 'SplitInIcePulses' in '-'.join(config['masks']):
        config['cleaning'] = 'SplitInIcePulses'
        config['cleaning_length'] = 'split_in_ice_pulses_event_length'

    if config['dev_run']:
        sets['train'] = sets['train'][0:20000]
        sets['val'] = sets['val'][0:20000]

    dataset = Dl(
        sets['train'],
        config,
        train_set,
        test=False
    )
    dataset = Dl(
        sets['val'],
        config,
        val_set,
        test=True
    )

    loss = torch.nn.MSELoss()
    reporter = Reporter(config, experiment_name)
    saver = Saver(config, dirs)

    Model = getattr(importlib.import_module('src.modules.' + config['model']), 'Model')

    model = Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['min_learning_rate'])

    trainer = Trainer(
        config,
        model,
        optimizer,
        loss,
        reporter,
        saver,
        inferer,
        train_dataset,
        val_dataset
    )

    trainer.fit()
    
    print('{}: Script done!'.format(get_time()))

if __name__ == '__main__':
    main()
