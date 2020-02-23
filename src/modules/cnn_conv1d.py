import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import io
import slack
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.modules.pickle_generator import PickleGenerator
from src.modules.resolution_comparison import ResolutionComparison
from src.modules.invert_transforms import TransformsInverter
from src.modules.losses import logcosh_loss
from src.modules.utils import get_time
from src.modules.utils import get_project_root


class CnnSystemConv1d(pl.LightningModule):
    def __init__(self, sets, config, files_and_dirs, val_freq, wandb, hparams, val_check_interval, experiment_name):
        super(CnnSystemConv1d, self).__init__()
        self.sets = sets
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.val_freq = val_freq
        self.wandb = wandb
        self.hparams = hparams
        self.val_check_interval = val_check_interval
        self.experiment_name = experiment_name

        self.PROJECT_ROOT = get_project_root()
        self.RUN_ROOT = self.PROJECT_ROOT.joinpath('runs')
        self.RUN_ROOT.mkdir(exist_ok=True)
        self.RUN_ROOT = self.RUN_ROOT.joinpath(self.experiment_name)
        self.RUN_ROOT.mkdir(exist_ok=True)

        self.train_loss = []
        self.train_batches_per_second = []
        self.val_batches_per_second = []
        self.train_true_energy = []
        self.train_event_length = []
        self.first_val = False
        self.first_train = True
        self.first_test = True

        self.column_names = [
            'file_number',
            'energy',
            'event_length'
        ]
        self.column_names += ['opponent_' + name for name in self.config.comparison_metrics]
        self.column_names += ['own_' + name.replace('true_', '') for name in self.config.targets]
        self.column_names += [name for name in self.config.targets]
        self.data = {name: [] for name in self.column_names}

        self.transform_object = TransformsInverter(self.config, self.files_and_dirs)

        if not self.config.dev_run:
            self.slack_token = os.environ["SLACK_API_TOKEN"]
            self.client = slack.WebClient(token=self.slack_token)

        self.comparisonclass = ResolutionComparison(self.wandb, self.config, self.experiment_name)

        self.conv1 = torch.nn.Conv1d(
            in_channels=len(self.config.features),
            out_channels=32,
            kernel_size=3
        )
        self.batchnorm1 = torch.nn.BatchNorm1d(
            num_features=32
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3
        )
        self.batchnorm2 = torch.nn.BatchNorm1d(
            num_features=64
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3
        )
        self.batchnorm3 = torch.nn.BatchNorm1d(
            num_features=128
        )
        self.conv4 = torch.nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.batchnorm4 = torch.nn.BatchNorm1d(
            num_features=256
        )
        self.linear1 = torch.nn.Linear(
            in_features=2560,
            out_features=2048
        )
        self.batchnorm5 = torch.nn.BatchNorm1d(
            num_features=2048
        )
        self.linear2 = torch.nn.Linear(
            in_features=2048,
            out_features=1024
        )
        self.batchnorm6 = torch.nn.BatchNorm1d(
            num_features=1024
        )
        self.linear3 = torch.nn.Linear(
            in_features=1024,
            out_features=len(self.config.targets)
        )


    def forward(self, x):
        x = F.max_pool1d(F.leaky_relu(self.conv1(x)), 2)
        x = self.batchnorm1(x)
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = self.batchnorm2(x)
        x = F.max_pool1d(F.leaky_relu(self.conv3(x)), 2)
        x = self.batchnorm3(x)
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)
        x = self.batchnorm4(x)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = F.leaky_relu(self.linear1(x))
        x = self.batchnorm5(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.batchnorm6(x)
        x = F.dropout(x, p=0.5)
        x = self.linear3(x)
        return x

    def on_epoch_start(self):
        print(
            '''
{}: Beginning epoch {}
            '''
            .format(get_time(), self.current_epoch)
        )
        if not self.config.dev_run:
            self.client.chat_postMessage(
                channel='training',
                text='Epoch {} begun.'.format(self.current_epoch)
            )


    def training_step(self, batch, batch_idx):
        if self.first_train:
            self.train_step = 1
            self.train_time_start = datetime.now()
            self.first_train = False
            self.first_val = True
        else:
            self.train_step += 1
        x, y, true_energy, event_length = batch
        if self.config.save_train_dists:
            self.train_true_energy.extend(true_energy.cpu().numpy())
            self.train_event_length.extend(event_length.cpu().numpy())
        y_hat = self.forward(x)
        # loss = F.mse_loss(y_hat, y)
        loss = logcosh_loss(y_hat, y)
        self.train_loss.append(loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if self.first_val and self.global_step != 0:
            self.train_time_end = datetime.now()
            self.train_time_delta = (self.train_time_end - self.train_time_start).total_seconds()
            self.val_time_start = datetime.now()
            self.first_val = False
            self.val_step = 1
        elif self.global_step != 0:
            self.val_step += 1
        x, y, true_energy, event_length = batch
        y_hat = self.forward(x)
        # loss = F.mse_loss(y_hat, y)
        loss = logcosh_loss(y_hat, y)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output = {
            'progress_bar': avg_loss,
            'log': {'val_loss': avg_loss}
        }
        if self.global_step != 0:
            avg_train_loss = torch.stack(self.train_loss).mean()
            self.train_loss = []
            self.val_time_end = datetime.now()
            self.val_time_delta = (self.val_time_end - self.val_time_start).total_seconds()
            self.first_train = True
            if self.config.wandb:
                metrics = {'train_loss': avg_train_loss, 'val_loss': avg_loss}
                self.wandb.log(metrics, step=self.global_step)
            print('''
{}: Step {} / epoch {}
          Train loss: {:.3f} / {:.1f} events/s
          Val loss:   {:.3f} / {:.1f} events/s
                '''
                .format(
                    get_time(),
                    self.global_step,
                    self.current_epoch,
                    avg_train_loss,
                    self.train_step * self.config.batch_size / self.train_time_delta,
                    avg_loss,
                    self.val_step * self.config.val_batch_size / self.val_time_delta
                )
            )
        self.train_batches_per_second = []
        self.val_batches_per_second = []
        return {'val_loss': avg_loss}

    def on_epoch_end(self):
        if not self.config.dev_run:
            self.client.chat_postMessage(
                channel='training',
                text='Epoch {} done'.format(self.current_epoch)
            )

    def test_step(self, batch, batch_nb):
        if self.first_test:
            print('{}: Testing started'.format(get_time()))
            if not self.config.dev_run:
                self.client.chat_postMessage(
                    channel='training',
                    text='Testing started.'
                )
            self.test_time_start = datetime.now()
            self.test_step_i = 1
            self.first_test = False
        else:
            self.test_step_i += 1
        x, y, comparisons, energy, event_length, file_number = batch
        y_hat = self.forward(x)
        # loss = F.mse_loss(y_hat, y)
        loss = logcosh_loss(y_hat, y)
        # transformed_y, transformed_y_hat = self.transform_object.transform_inversion(y, y_hat)
        values = [
            list(file_number),
            energy.tolist(),
            event_length.tolist(),
            *[comparison.tolist() for comparison in comparisons],
            *[y_hat[:, i].tolist() for i in range(y_hat.size(1))],
            *[y[:, i].tolist() for i in range(y.size(1))]
        ]
        for i, key in enumerate(self.data):
            self.data[key].extend(values[i])
        return {'test_loss': loss}

    def test_end(self, outputs):
        self.test_time_end = datetime.now()
        self.test_time_delta = (self.test_time_end - self.test_time_start).total_seconds()
        print(
            '{}: Testing ended, {:.1f} events/s'.format(
                get_time(),
                self.test_step_i * self.config.val_batch_size / self.test_time_delta
            )
        )
        if not self.config.dev_run:
            self.client.chat_postMessage(
                channel='training',
                text='Testing ended.'
            )
        comparison_df = pd.DataFrame().from_dict(self.data)
        print('{}: Saving predictions file'.format(get_time()))
        file_name = self.RUN_ROOT.joinpath('comparison_dataframe_parquet.gzip')
        comparison_df.to_parquet(
            str(file_name),
            compression='gzip'
        )
        if self.config.wandb:
            print('{}: Uploading comparison df to wandb'.format(get_time()))
            self.wandb.save(str(file_name))
        self.comparisonclass.testing_ended(file_name, self.RUN_ROOT, self.train_true_energy, self.train_event_length)
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.min_learning_rate,
            # weight_decay=0.9,
        )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer,
        #     gamma=0.99
        # )
        # return [optimizer], [scheduler]
        return optimizer

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_i,
        second_order_closure=None
    ):
        if self.trainer.global_step < (self.train_batches + self.val_batches):
            lr_scale = min(1., float(self.trainer.global_step + 1) / (self.train_batches + self.val_batches))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.config.max_learning_rate
        else:
            lr_scale = 0.999999
            for pg in optimizer.param_groups:
                if pg['lr'] >= self.config.min_learning_rate:
                    pg['lr'] = lr_scale * pg['lr']
                else:
                    pg['lr'] = pg['lr']
        optimizer.step()   
        optimizer.zero_grad()
        if self.config.wandb == True:
            self.wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=self.global_step)

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataset = PickleGenerator(
            self.config,
            self.sets['train'],
            test=False,
            conv_type='conv1d'
        )
        dl = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True
        )
        no_of_samples = len(self.train_dataset)
        self.train_batches = np.floor(no_of_samples / self.config.batch_size)
        print('No. of train samples:', no_of_samples)
        return dl

    @pl.data_loader
    def val_dataloader(self):
        self.val_dataset = PickleGenerator(
            self.config,
            self.sets['val'],
            test=False,
            conv_type='conv1d'
        ) 
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            drop_last=True
        )
        no_of_samples = len(self.val_dataset)
        self.val_batches = np.floor(no_of_samples / self.config.val_batch_size)
        print('No. of validation samples:', no_of_samples)
        return dl

    @pl.data_loader
    def test_dataloader(self):
        self.test_dataset = PickleGenerator(
            self.config,
            self.sets['val'],
            test=True,
            conv_type='conv1d'
        )
        dl = DataLoader(
            self.test_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            drop_last=True
        )
        no_of_samples = len(self.test_dataset)
        print('No. of test samples:', no_of_samples)
        return dl
