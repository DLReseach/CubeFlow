import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data_loader.cnn_generator import CnnGenerator
from loggers.loggers import WandbLogger


class CnnSystem(pl.LightningModule):
    def __init__(self, sets, config, wandb):
        super(CnnSystem, self).__init__()
        self.sets = sets
        self.config = config
        self.wandb = wandb
        self.logclass = WandbLogger(wandb)
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
            in_features=2560,
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
        if self.config.wandb == True:
            self.logclass.reset()
        else:
            pass


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        if self.config.wandb == True:
            self.logclass.update({'train_loss': loss})
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        if self.config.wandb == True:
            self.logclass.update({'val_loss': loss})
        return {'val_loss': loss}


    def on_epoch_end(self):
        if self.config.wandb == True:
            self.logclass.log_metrics()
        else:
            pass


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.0001)


    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            CnnGenerator(self.config, self.sets['train'], test=False),
            batch_size=None,
            num_workers=self.config.num_workers
        )


    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            CnnGenerator(self.config, self.sets['validate'], test=False),
            batch_size=None,
            num_workers=self.config.num_workers
        )


    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            CnnGenerator(self.config, self.sets['test'], test=True),
            batch_size=None,
            num_workers=self.config.num_workers
        )
