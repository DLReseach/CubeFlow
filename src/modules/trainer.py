import torch
import numpy as np

from src.modules.utils import get_time


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        loss,
        reporter,
        saver,
        train_dataset,
        val_dataset
    ):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.reporter = reporter
        self.saver = saver
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')        
        self.model.to(self.device)

        self.global_step = 0

        self.epoch_validation_loss = []

    def train_epoch(self):
        self.model.train()
        self.train_dataset.on_epoch_start()
        for self.i, batch in enumerate(self.train_dl):
            self.reporter.on_training_batch_start()
            x = batch[0].to(self.device).float()
            y = batch[1].to(self.device).float()
            y_hat = self.model.forward(x)
            loss = self.loss(y_hat, y)
            loss.backward()
            self.reporter.on_training_batch_end(loss)
            self.optimizer_step(self.optimizer)
            if self.i % self.train_length == 0 and self.i > 0:
                self.reporter.on_intermediate_training_end()
                self.intermediate_validation()

    def intermediate_validation(self):
        self.model.eval()
        self.val_dataset.on_epoch_start()
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                self.reporter.on_intermediate_validation_batch_start()
                x = batch[0].to(self.device).float()
                y = batch[1].to(self.device).float()
                y_hat = self.model.forward(x)
                loss = self.loss(y_hat, y)
                self.reporter.on_intermediate_validation_batch_end(loss)
                if i == self.val_length:
                    break
            self.reporter.on_intermediate_validation_end()
        self.model.train()

    def epoch_validation(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_dl:
                self.reporter.on_epoch_validation_batch_start()
                x = batch[0].to(self.device).float()
                y = batch[1].to(self.device).float()
                y_hat = self.model.forward(x)
                loss = self.loss(y_hat, y)
                self.reporter.on_epoch_validation_batch_end(loss)
            epoch_val_loss = self.reporter.on_epoch_validation_end()
            self.model.train()
        return epoch_val_loss

    def optimizer_step(
        self,
        optimizer
    ):
        if self.global_step < (self.train_batches + self.val_batches):
            lr_scale = min(1., float(self.global_step + 1) / (self.train_batches + self.val_batches))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.config['max_learning_rate']
        elif self.i == 0:
            for pg in optimizer.param_groups:
                pg['lr'] = self.config['max_learning_rate'] / (self.epoch * 1)
        optimizer.step()   
        optimizer.zero_grad()
        self.reporter.optimizer_step(optimizer.param_groups[0]['lr'])
        self.global_step += 1


    def fit(self):
        self.create_dataloaders()
        for self.epoch in range(self.config['num_epochs']):
            self.reporter.on_epoch_start()
            self.train_epoch()
            epoch_val_loss = self.epoch_validation()
            make_early_stop = self.saver.early_stopping(self.epoch, epoch_val_loss, self.model.state_dict(), self.optimizer.state_dict())
            self.reporter.on_epoch_end()
            if make_early_stop:
                print('{}: early stopping activated'.format(get_time()))
                break
        self.saver.upload_model_files()
        

    def create_dataloaders(self):
        self.train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.config['num_workers'],
            shuffle=False
        )
        no_of_samples = len(self.train_dataset) * self.config['batch_size']
        self.train_batches = len(self.train_dataset)
        print('No. of train samples:', no_of_samples)
        self.val_dl = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=None,
            num_workers=self.config['num_workers'],
            shuffle=False
        )
        no_of_samples = len(self.val_dataset) * self.config['batch_size']
        self.val_batches = len(self.val_dataset)
        print('No. of validation samples:', no_of_samples)
        self.train_length = int(self.config['val_check_frequency'] * len(self.train_dl))
        self.val_length = int(self.config['val_check_frequency'] * len(self.val_dl))
        if self.val_length == 0:
            self.val_length = len(self.val_dl)