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
        train_dl,
        val_dl
    ):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.reporter = reporter
        self.saver = saver
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_length = int(self.config.val_check_frequency * len(self.train_dl))
        self.val_length = int(self.config.val_check_frequency * len(self.val_dl))
        if self.val_length == 0:
            self.val_length = len(self.val_dl)

    def train_steps(self, train_dl_iter):
        self.model.train()
        for i in range(len(self.train_dl)):
            self.reporter.training_batch_start()
            x, y, train_true_energy, train_event_length = next(train_dl_iter)
            self.optimizer.zero_grad()
            y_hat = self.model.forward(x)
            loss = self.loss(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.saver.train_step(train_true_energy, train_event_length)
            self.reporter.training_batch_end(loss)
            if i > 0 and i % self.train_length == 0:
                val_dl_iter = iter(self.val_dl)
                self.val_steps(val_dl_iter)
                self.saver.on_val_end()
                _ = self.reporter.on_val_end()
                self.model.train()

    def val_steps(self, val_dl_iter):
        self.model.eval()
        with torch.no_grad():
            for i in range(self.val_length):
                self.reporter.val_batch_start()
                x, y, comparisons, energy, event_length, file_number = next(val_dl_iter)
                y_hat = self.model.forward(x)
                loss = self.loss(y_hat, y)
                self.saver.on_val_step(x, y, y_hat, comparisons, energy, event_length, file_number)
                self.reporter.val_batch_end(loss)

    def fit(self):
        for epoch in range(self.config.num_epochs):
            train_dl_iter = iter(self.train_dl)
            self.reporter.on_epoch_start()
            self.train_steps(train_dl_iter)