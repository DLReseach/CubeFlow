import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import wandb as wandb
import numpy as np
import matplotlib.cbook
import warnings
import joblib
from argparse import Namespace
# from torch_lr_finder import LRFinder

from src.lightning_systems.cnn import CnnSystem
from preprocessing.cnn_preprocessing import CnnPreprocess
from utils.config import process_config
from utils.utils import get_args
from utils.utils import get_project_root
from utils.utils import get_time
from utils.utils import print_data_set_sizes
from utils.utils import create_experiment_name
from utils.utils import set_random_seed
from utils.utils import get_files_and_dirs
from preprocessing.mask_and_split import MaskAndSplit
from utils.math_funcs import angle_between
from plots.plot_functions import histogram

warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print('missing or invalid arguments')
        exit(0)

    experiment_name = create_experiment_name(config, slug_length=2)

    if config.wandb and config.exp_name != 'lr_finder':
        wandb.init(
                project='cubeflow',
                name=experiment_name
            )

    hparams = Namespace(**{'learning_rate': config.max_learning_rate})

    files_and_dirs = get_files_and_dirs(config)
    mask_and_split = MaskAndSplit(config, files_and_dirs)
    sets = mask_and_split.split()
    val_check_interval = int(config.val_check_frequency * len(sets['train']) / config.batch_size)
    model = CnnSystem(sets, config, files_and_dirs, val_check_interval, wandb, hparams, val_check_interval)
    
    if config.wandb:
        wandb.watch(model, log='gradients')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=config.patience,
        verbose=False,
        mode='min'
    )

    if config.gpulab:
        gpus = config.gpulab_gpus
    else:
        gpus = config.gpus

    trainer = Trainer(
        show_progress_bar=False,
        gpus=gpus,
        max_epochs=config.num_epochs,
        fast_dev_run=config.dev_run,
        early_stop_callback=None if config.patience == 0 else early_stop_callback,
        val_check_interval=val_check_interval
    )
    trainer.fit(model)

    if config.test:
        trainer.test()

if __name__ == '__main__':
    main()
