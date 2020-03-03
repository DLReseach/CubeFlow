import torch
import numpy as np


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        loss,
        reporter,
        saver,
        inferer,
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
        self.inferer = inferer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model.to(self.device)

        self.global_step = 0

        self.epoch_validation_loss = []

    def train_epoch(self, train_dl_iter):
        self.model.train()
        for i in range(len(self.train_dl)):
            self.reporter.on_training_batch_start()
            x, y, train_true_energy, train_event_length = next(train_dl_iter)
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model.forward(x)
            loss = self.loss(y_hat, y)
            loss.backward()
            self.optimizer_step(self.optimizer)
            self.saver.train_step(train_true_energy, train_event_length)
            if i > 0 and i % self.train_length == 0:
                self.reporter.on_training_batch_end(loss)
                self.reporter.on_intermediate_training_end()
                self.intermediate_validation()

    def intermediate_validation(self):
        self.val_dataset.shuffle()
        val_dl_iter = iter(self.val_dl)
        self.model.eval()
        with torch.no_grad():
            for i in range(self.val_length):
                self.reporter.on_intermediate_validation_batch_start()
                x, y, comparisons, energy, event_length, file_number = next(val_dl_iter)
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model.forward(x)
                loss = self.loss(y_hat, y)
                self.reporter.on_intermediate_validation_batch_end(loss)
            self.reporter.on_intermediate_validation_end()
        self.model.train()

    def epoch_validation(self):
        val_dl_iter = iter(self.val_dl)
        self.model.eval()
        with torch.no_grad():
            for i in range(len(val_dl_iter)):
                self.reporter.on_epoch_validation_batch_start()
                x, y, comparisons, energy, event_length, file_number = next(val_dl_iter)
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.model.forward(x)
                loss = self.loss(y_hat, y)
                self.reporter.on_epoch_validation_batch_end(loss)
            self.epoch_validation_loss.append(self.reporter.on_epoch_validation_end())
            self.model.train()

    def optimizer_step(
        self,
        optimizer
    ):
        if self.global_step < (self.train_batches + self.val_batches):
            lr_scale = min(1., float(self.global_step + 1) / (self.train_batches + self.val_batches))
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
        self.global_step += 1


    def fit(self):
        self.create_dataloaders()
        for epoch in range(self.config.num_epochs):
            self.train_dataset.shuffle()
            train_dl_iter = iter(self.train_dl)
            self.reporter.on_epoch_start()
            self.train_epoch(train_dl_iter)
            self.epoch_validation()
            self.saver.save_model_state(epoch, self.model.state_dict(), self.optimizer.state_dict())
            self.reporter.on_epoch_end()

    def create_dataloaders(self):
        self.train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            shuffle=False
        )
        no_of_samples = len(self.train_dataset)
        self.train_batches = np.floor(no_of_samples / self.config.batch_size)
        print('No. of train samples:', no_of_samples)
        self.val_dl = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
            shuffle=False
        )
        no_of_samples = len(self.val_dataset)
        self.val_batches = np.floor(no_of_samples / self.config.val_batch_size)
        print('No. of validation samples:', no_of_samples)
        self.train_length = int(self.config.val_check_frequency * len(self.train_dl))
        self.val_length = int(self.config.val_check_frequency * len(self.val_dl))
        if self.val_length == 0:
            self.val_length = len(self.val_dl)