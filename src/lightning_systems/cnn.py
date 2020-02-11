import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import io
import numpy as np
from datetime import datetime
from data_loader.pickle_generator import PickleGenerator
from metrics.resolution_comparison import ResolutionComparison
from transforms.invert_transforms import TransformsInverter


class CnnSystem(pl.LightningModule):
    def __init__(self, sets, config, files_and_dirs, val_freq, wandb, hparams, val_check_interval):
        super(CnnSystem, self).__init__()
        self.sets = sets
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.val_freq = val_freq
        self.wandb = wandb
        self.hparams = hparams
        self.val_check_interval = val_check_interval
        self.train_loss = []
        self.train_batches_per_second = []
        self.val_batches_per_second = []
        self.first_val = False
        self.first_train = True
        self.comparisonclass = ResolutionComparison(self.wandb, self.config)

        self.conv1 = torch.nn.Conv1d(
            in_channels=len(self.config.features),
            out_channels=32,
            kernel_size=5
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5
        )
        self.conv3 = torch.nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5
        )
        self.conv4 = torch.nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=5
        )
        self.linear1 = torch.nn.Linear(
            in_features=11264,
            out_features=len(self.config.targets)
        )


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 2)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)
        x = torch.flatten(x, start_dim=1, end_dim=2)
        x = self.linear1(x)
        return x

    def on_epoch_start(self):
        print(
            '''
{}: Begining epoch {}
            '''
            .format(datetime.now().strftime('%H:%M:%S'), self.current_epoch + 1)
        )

    def training_step(self, batch, batch_idx):
        if self.first_train:
            self.train_time_start = datetime.now()
            self.first_train = False
            self.first_val = True
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.train_loss.append(loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        if self.first_val and self.global_step != 0:
            self.train_time_end = datetime.now()
            self.train_time_delta = (self.train_time_end - self.train_time_start).total_seconds()
            self.val_time_start = datetime.now()
            self.first_val = False
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
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
          Train loss: {:.3f} / {:.1f} batches/s
          Val loss:   {:.3f} / {:.1f} batches/s
                '''
                .format(
                    datetime.now().strftime('%H:%M:%S'),
                    self.global_step,
                    self.current_epoch + 1,
                    avg_train_loss,
                    self.val_check_interval / self.train_time_delta,
                    avg_loss,
                    self.val_batches / self.val_time_delta
                )
            )
        self.train_batches_per_second = []
        self.val_batches_per_second = []
        return {'val_loss': avg_loss}

    def test_step(self, batch, batch_nb):
        x, y, comparisons, energy = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        transform_object = TransformsInverter(y, y_hat, self.config, self.files_and_dirs)
        transformed_y, transformed_y_hat = transform_object.transform_inversion()
        self.comparisonclass.update_values(transformed_y_hat, transformed_y, comparisons, energy)
        return {'test_loss': loss}

    def test_end(self, outputs):
        self.comparisonclass.testing_ended()
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.min_learning_rate
        )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer,
        #     gamma=0.9
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
            for pg in optimizer.param_groups:
                pg['lr'] = 0.99999 * pg['lr']
        optimizer.step()   
        optimizer.zero_grad()
        if self.config.wandb == True:
            self.wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=self.global_step)

    @pl.data_loader
    def train_dataloader(self):
        self.train_dataset = PickleGenerator(
            self.config,
            self.sets['train'],
            test=False
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
            test=False
        ) 
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True
        )
        no_of_samples = len(self.val_dataset)
        self.val_batches = np.floor(no_of_samples / self.config.batch_size)
        print('No. of validation samples:', no_of_samples)
        return dl

    @pl.data_loader
    def test_dataloader(self):
        self.test_dataset = PickleGenerator(
            self.config,
            self.sets['val'],
            test=True
        )
        dl = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True
        )
        no_of_samples = len(self.test_dataset)
        print('No. of test samples:', no_of_samples)
        return dl
