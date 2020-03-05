import torch
from pathlib import Path
import importlib
import wandb
import json
from bunch import Bunch
import os

from src.modules.utils import get_time
from src.modules.utils import get_project_root
from src.modules.utils import get_files_and_dirs
from src.modules.mask_and_split import MaskAndSplit
from src.modules.reporter import Reporter
from src.modules.saver import Saver
from src.modules.resolution_comparison import ResolutionComparison
from src.modules.pickle_generator import PickleGenerator
from src.modules.trainer import Trainer
from src.modules.inferer import Inferer

RUN_NAME = 'lemon-akita'
ROOT = get_project_root()
DOWNLOAD_FOLDER = ROOT.joinpath('mains/downloads').joinpath(RUN_NAME)
DOWNLOAD_FOLDER.mkdir(exist_ok=True)

api = wandb.Api()

runs = api.runs('ehrhorn/cubeflow')
for run in runs:
    if run.name == RUN_NAME:
        run_id = run.id

run = api.run('ehrhorn/cubeflow/' + run_id)
print('{}: Downloading run data'.format(get_time()))
for file in run.files():
    if file.name == 'model.pt' or file.name.split('.')[-1] == 'py' or file.name == 'cnn.json':
        if file.name != 'code/mains/cnn.py':
            file.download(replace=True, root=str(DOWNLOAD_FOLDER))
            if file.name != 'model.pt':
                model_file_name = file.name.split('/')[-1].split('.')[0]

JSON_FILE = DOWNLOAD_FOLDER.joinpath('cnn.json')
with open(str(JSON_FILE), 'r') as config_file:
    config_dict = json.load(config_file)
config = Bunch(config_dict)
config.save_train_dists = False
config.wandb = False

Model = getattr(importlib.import_module('downloads.' + RUN_NAME + '.' + model_file_name), 'Model')

files_and_dirs = get_files_and_dirs(config, RUN_NAME)
mask_and_split = MaskAndSplit(config, files_and_dirs)
sets = mask_and_split.split()

test_dataset = PickleGenerator(
    config,
    sets['test'],
    test=True,
    val=False,
    conv_type='conv1d'
) 

comparer_config = {
    'dom_plots': False,
    'use_train_dists': True,
    'only_use_metrics': None,
    'legends': True,
    'use_own': True,
    'reso_hists': False,
    'wandb': None
}

loss = torch.nn.MSELoss()
reporter = Reporter(config, wandb, None, RUN_NAME)
saver = Saver(config, None, files_and_dirs)
comparer = ResolutionComparison(config.comparison_metrics, files_and_dirs, comparer_config, reporter)

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=config.min_learning_rate)

inferer = Inferer(model, optimizer, loss, test_dataset, saver, config)
print('{}: Beginning inference'.format(get_time()))
inferer.infer(str(DOWNLOAD_FOLDER.joinpath('model.pt')))
print('{}: Beginning comparison'.format(get_time()))
comparer.testing_ended()
print('{}: Script done!'.format(get_time()))
