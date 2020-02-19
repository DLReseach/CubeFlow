import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import wandb as wandb
import matplotlib.cbook
import warnings
from argparse import Namespace
import slack
# from torch_lr_finder import LRFinder

from lightning_systems.cnn_conv1d import CnnSystemConv1d
from lightning_systems.cnn_conv2d import CnnSystemConv2d
from utils.config import process_config
from utils.utils import get_args
from utils.utils import create_experiment_name
from utils.utils import get_files_and_dirs
from preprocessing.mask_and_split import MaskAndSplit

warnings.filterwarnings(
    'ignore',
    category=matplotlib.cbook.mplDeprecation
)

slack_token = os.environ["SLACK_API_TOKEN"]
client = slack.WebClient(token=slack_token)


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except Exception:
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
    val_check_interval = int(
        config.val_check_frequency * len(sets['train']) / config.batch_size
    )
    model = CnnSystemConv1d(
        sets,
        config,
        files_and_dirs,
        val_check_interval,
        wandb,
        hparams,
        val_check_interval
    )

    if config.wandb:
        wandb.watch(model, log='gradients')

    # if config.patience == 0:
    #     early_stop_callback = None
    # else:
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=config.patience,
        verbose=True,
        mode='min'
    )

    if config.gpulab:
        gpus = config.gpulab_gpus
        use_amp = False
        if len(gpus) > 1:
            distributed_backend = 'dp'
        else:
            distributed_backend = None
    else:
        gpus = config.gpus
        if gpus > 0:
            use_amp = False
            if gpus > 1:
                distributed_backend = 'dp'
            else:
                distributed_backend = None
        else:
            use_amp = False
            distributed_backend = None

    trainer = Trainer(
        show_progress_bar=False,
        gpus=gpus,
        max_epochs=config.num_epochs,
        fast_dev_run=config.dev_run,
        early_stop_callback=early_stop_callback,
        val_check_interval=val_check_interval,
        use_amp=use_amp,
        distributed_backend=distributed_backend
    )

    trainer.fit(model)

    if config.test:
        trainer.test()

    client.chat_postMessage(
        channel='training',
        text='Script done.'
    )

if __name__ == '__main__':
    main()
