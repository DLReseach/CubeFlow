from pathlib import Path
import time
import datetime
from coolname import generate_slug
import json


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def get_time():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    return st


def get_date_and_time():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return st


def create_experiment_name(slug_length):
    cool_name = generate_slug(2)
    # today = str(datetime.date.today())
    # experiment_name = config.exp_name + '_' + today + '.' + cool_name
    return cool_name


def get_dirs_and_config(experiment_name, train):
    files_and_dirs = {}
    data_path = Path().home().joinpath('CubeFlowData')
    files_and_dirs['transformers'] = data_path.joinpath('transformers')
    files_and_dirs['masks'] = data_path.joinpath('masks')
    files_and_dirs['run'] = data_path.joinpath('runs').joinpath(experiment_name)
    files_and_dirs['run'].mkdir(exist_ok=True, parents=True)
    files_and_dirs['train_distributions'] = data_path.joinpath('train_distributions')
    files_and_dirs['project'] = get_project_root()
    if train:
        with open(files_and_dirs['project'].joinpath('configs').joinpath('config.json'), 'r') as f:
            config = json.load(f)
        first_run = False
    else:
        with open(files_and_dirs['run'].joinpath('config.json'), 'r') as f:
            config = json.load(f)
        mask_name = '-'.join(config['masks'])
        files_and_dirs['dbs'] = data_path.joinpath('dbs').joinpath(mask_name)
        if not files_and_dirs['dbs'].is_dir():
            files_and_dirs['dbs'].mkdir(exist_ok=False)
            first_run = True
        else:
            first_run = False
    return files_and_dirs, config, first_run
