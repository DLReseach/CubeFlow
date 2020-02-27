import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import wandb as wandb
import matplotlib.cbook
import warnings
from argparse import Namespace
import slack
# from torch_lr_finder import LRFinder

from src.modules.cnn_conv1d import CnnSystemConv1d
from src.modules.config import process_config
from src.modules.utils import get_args
from src.modules.utils import create_experiment_name
from src.modules.utils import get_files_and_dirs
from src.modules.utils import get_time
from src.modules.mask_and_split import MaskAndSplit
from src.modules.reporter import Reporter
from src.modules.saver import Saver
from src.modules.resolution_comparison import ResolutionComparison

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

    if config.dev_run:
        config.train_fraction = 0.001
        config.val_fraction = 0.001
        config.test_fraction = 0.001

    hparams = Namespace(**{'learning_rate': config.max_learning_rate})

    files_and_dirs = get_files_and_dirs(config, experiment_name)
    mask_and_split = MaskAndSplit(config, files_and_dirs)
    sets = mask_and_split.split()
    val_check_interval = int(
        config.val_check_frequency * len(sets['train']) / config.batch_size
    )

    comparer_config = {
        'dom_plots': False,
        'use_train_dists': True,
        'only_use_metrics': None,
        'legends': True,
        'use_own': True,
        'reso_hists': False,
        'wandb': config.wandb
    }

    reporter = Reporter(config, wandb, client)
    saver = Saver(config, wandb, files_and_dirs)
    comparer = ResolutionComparison(config.comparison_metrics, files_and_dirs, comparer_config, reporter)

    model = CnnSystemConv1d(
        sets,
        config,
        files_and_dirs,
        val_check_interval,
        wandb,
        hparams,
        val_check_interval,
        experiment_name,
        reporter,
        saver
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
        early_stop_callback=early_stop_callback,
        val_check_interval=val_check_interval,
        use_amp=use_amp,
        distributed_backend=distributed_backend,
        num_sanity_val_steps=0
    )

    trainer.fit(model)

    if config.wandb:
        comp_df = files_and_dirs['run_root'].joinpath(
            'comparison_dataframe_parquet.gzip'
        )
        wandb.save(str(comp_df))

    comparer.testing_ended()

    # if config.test:
    #     trainer.test()

    if not config.dev_run:
        client.chat_postMessage(
            channel='training',
            text='Script done.'
        )
    
    print('{}: Script done!'.format(get_time()))

if __name__ == '__main__':
    main()
