import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from warmup_scheduler import GradualWarmupScheduler
import math
from data_loader.pickle_generator import PickleGenerator
from loggers.loggers import WandbLogger
from metrics.comparison import RetroCrsComparison


class CnnSystem(pl.LightningModule):
    def __init__(self, sets, config, wandb):
        super(CnnSystem, self).__init__()
        self.sets = sets
        self.config = config
        self.wandb = wandb
        self.logclass = WandbLogger(self.wandb, ['train_loss', 'val_loss'])
        self.comparisonclass = RetroCrsComparison(self.wandb, self.config)

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
        if self.config.wandb == True:
            self.logclass.reset()
        else:
            pass


    def training_step(self, batch, batch_idx):
        x, y, comparisons, energy = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        if self.config.wandb == True:
            self.logclass.update({'train_loss': loss})
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        x, y, comparisons, energy = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        if self.config.wandb == True:
            self.logclass.update({'val_loss': loss})
        return {'val_loss': loss}


    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output = {
            'progress_bar': avg_loss,
            'log': {'val_loss': avg_loss}
        }
        return {'val_loss': avg_loss}


    def test_step(self, batch, batch_nb):
        x, y, comparisons, energy = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        # x_test, y_test = self.test_dataset[batch_nb]
        # assert torch.all(x_test.eq(x)), 'Whoops, x and x_test are not the same'
        # assert torch.all(y_test.eq(y)), 'Whoops, y and y_test are not the same'
        self.comparisonclass.update_values(y_hat, y, comparisons, energy)
        return {'test_loss': loss}


    def test_end(self, outputs):
        self.comparisonclass.testing_ended()
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': avg_loss}


    def on_epoch_end(self):
        if self.config.wandb == True:
            self.logclass.log_metrics()
        else:
            pass


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.min_learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.1
        )
        scheduler_warmup = GradualWarmupScheduler(
            optimizer,
            multiplier=8,
            total_epoch=10,
            after_scheduler=scheduler
        )
        return [optimizer], [scheduler_warmup]


    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_i,
        second_order_closure=None
    ):
        optimizer.step()   
        optimizer.zero_grad()
        if batch_nb == 0 and self.config.wandb == True:
            self.wandb.log({'learning_rate': optimizer.param_groups[0]['lr']})
        

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
            num_workers=self.config.num_workers
        )
        no_of_samples = len(self.train_dataset)
        print('No. of train samples:', no_of_samples)
        return dl


    @pl.data_loader
    def val_dataloader(self):
        self.val_dataset = PickleGenerator(
            self.config,
            self.sets['validate'],
            test=False
        ) 
        dl = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        no_of_samples = len(self.val_dataset)
        print('No. of validation samples:', no_of_samples)
        return dl


    @pl.data_loader
    def test_dataloader(self):
        self.test_dataset = PickleGenerator(
            self.config,
            self.sets['test'],
            test=True
        )
        dl = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=0
        )
        no_of_samples = len(self.test_dataset)
        print('No. of test samples:', no_of_samples)
        return dl
