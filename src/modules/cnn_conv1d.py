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
from src.modules.losses import logcosh_loss
from src.modules.utils import get_time
from src.modules.utils import get_project_root


class CnnSystemConv1d(pl.LightningModule):
    def __init__(
        self,
        sets,
        config,
        files_and_dirs,
        val_freq,
        wandb,
        hparams,
        val_check_interval,
        experiment_name,
        reporter,
        saver
    ):
        super(CnnSystemConv1d, self).__init__()
        self.sets = sets
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.val_freq = val_freq
        self.wandb = wandb
        self.hparams = hparams
        self.val_check_interval = val_check_interval
        self.experiment_name = experiment_name
        self.reporter = reporter
        self.saver = saver

        self.PROJECT_ROOT = get_project_root()
        self.RUN_ROOT = self.PROJECT_ROOT.joinpath('runs')
        self.RUN_ROOT.mkdir(exist_ok=True)
        self.RUN_ROOT = self.RUN_ROOT.joinpath(self.experiment_name)
        self.RUN_ROOT.mkdir(exist_ok=True)
        if self.config.save_train_dists:
            self.TRAIN_DISTS_PATH = self.PROJECT_ROOT.joinpath('train_distributions')
            self.TRAIN_DISTS_PATH.mkdir(exist_ok=True)
            self.files_and_dirs['train_dists_path'] = self.TRAIN_DISTS_PATH

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
            out_features=4096
        )
        self.batchnorm6 = torch.nn.BatchNorm1d(
            num_features=4096
        )
        self.linear3= torch.nn.Linear(
            in_features=4096,
            out_features=2048
        )
        self.batchnorm7 = torch.nn.BatchNorm1d(
            num_features=2048
        )
        self.linear4 = torch.nn.Linear(
            in_features=2048,
            out_features=1024
        )
        self.batchnorm8 = torch.nn.BatchNorm1d(
            num_features=1024
        )
        self.linear5 = torch.nn.Linear(
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
        x = F.leaky_relu(self.linear3(x))
        x = self.batchnorm7(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.batchnorm8(x)
        x = F.dropout(x, p=0.5)
        x = self.linear5(x)
        return x

    def on_epoch_start(self):
        self.reporter.on_epoch_start()

    def training_step(self, batch, batch_idx):
        self.reporter.training_batch_start()
        x, y, train_true_energy, train_event_length = batch
        y_hat = self.forward(x)
        # loss = F.mse_loss(y_hat, y)
        loss = logcosh_loss(y_hat, y)
        self.reporter.training_batch_end(
            loss
        )
        self.saver.train_step(train_true_energy, train_event_length)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        self.reporter.val_batch_start()
        x, y, comparisons, energy, event_length, file_number = batch
        y_hat = self.forward(x)
        # loss = F.mse_loss(y_hat, y)
        loss = logcosh_loss(y_hat, y)
        self.saver.on_val_step(x, y, y_hat, comparisons, energy, event_length, file_number)
        self.reporter.val_batch_end(loss)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        self.saver.on_val_end()
        avg_val_loss = self.reporter.on_val_end()
        return {'val_loss': avg_val_loss}

    def on_epoch_end(self):
        self.reporter.on_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.min_learning_rate
        )
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
        self.reporter.optimizer_step(optimizer.param_groups[0]['lr'])

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataset = PickleGenerator(
            self.config,
            self.sets['train'],
            test=False,
            val=False,
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
            val=True,
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
            val=False,
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
