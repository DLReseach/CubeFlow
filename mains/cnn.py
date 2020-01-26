import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import wandb as wandb
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings
import joblib
from warmup_scheduler import GradualWarmupScheduler
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
from utils.math_funcs import angle_between
from plots.plot_functions import histogram

warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device', device)


# @profile
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

    if config.wandb == True and config.exp_name != 'lr_finder':
        wandb.init(
                project='cubeflow',
                name=experiment_name
            )

    sets = joblib.load(
        get_project_root().joinpath(
            'sets/' + str(config.particle_type) + '.joblib'
        )
    )
    print('Starting preprocessing at {}'.format(get_time()))
    data = CnnPreprocess(sets, config)
    sets = data.return_indices()
    print('Ended preprocessing at {}'.format(get_time()))

    set_random_seed()

    # summary(model, input_size=(len(config.features), config.max_doms))

    model = CnnSystem(sets, config, wandb)

    if config.wandb == True:
        wandb.watch(model)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=config.patience,
        verbose=False,
        mode='min'
    )

    trainer = Trainer(
        gpus=config.gpus,
        max_epochs=config.num_epochs,
        # fast_dev_run=config.dev_run,
        early_stop_callback=None if config.patience == 0 else early_stop_callback
    )
    # trainer = Trainer(
    #     max_epochs=1,
    #     early_stop_callback=None
    # )
    trainer.fit(model)
    trainer.test()

    #     resolution = np.empty((0, len(config.targets)))
    #     direction = np.empty((0, 1))
    #     for inputs, targets in test_generator:
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #         predictions = prediction_step(model, inputs)
    #         resolution = np.vstack(
    #             [resolution, (targets.cpu() - predictions.cpu())]
    #         )
    #         for i in range(predictions.shape[0]):
    #             angle = angle_between(
    #                 targets.cpu()[i, :],
    #                 predictions.cpu()[i, :]
    #             )
    #             direction = np.vstack([direction, angle])

    #     if config.wandb == True:
    #         fig, ax = histogram(
    #             data=direction,
    #             title='arccos[y_truth . y_pred / (||y_truth|| ||y_pred||)]',
    #             xlabel='Angle (radians)',
    #             ylabel='Frequency',
    #             width_scale=1,
    #             bins='fd'
    #         )
    #         wandb.log({'Angle error': fig})

if __name__ == '__main__':
    main()
