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
from src.modules.pickle_dataloader import PickleDataset
from src.modules.losses import logcosh_loss
from src.modules.trainer import Trainer
from src.modules.inferer import Inferer

warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception:
        print('missing or invalid arguments')
        exit(0)

    experiment_name = 'pompous-puma'

    files_and_dirs = get_files_and_dirs(config, experiment_name)
    mask_and_split = MaskAndSplit(config, files_and_dirs)
    sets = mask_and_split.split()

    test_dataset = PickleDataset(
        sets['test'],
        config,
        'test'
    ) 

    loss = torch.nn.MSELoss()

    Model = getattr(importlib.import_module('src.modules.' + config.model), 'Model')

    model = Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.min_learning_rate)

    inferer = Inferer(model, optimizer, loss, test_dataset, config)

    model_path = files_and_dirs['run_root'].joinpath('model.pt')

    print('{}: Beginning inference'.format(get_time()))

    inferer.infer(model_path, files_and_dirs['run_root'])

    print('{}: Script done!'.format(get_time()))

if __name__ == '__main__':
    main()
