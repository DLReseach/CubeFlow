import torch
from datetime import datetime

from src.modules.utils import get_time


class Reporter:
    def __init__(self, config, wandb, client):
        super(Reporter, self).__init__()
        self.config = config

        self.current_epoch = 0
        self.train_loss = []
        self.training_step = 0
        self.train_time_delta = []
        self.val_loss = []
        self.val_step = 0
        self.val_time_delta = []
        self.train_true_energy = []
        self.train_event_length = []

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

    def on_epoch_end(self):
        if not self.config.dev_run:
            self.client.chat_postMessage(
                channel='training',
                text='Epoch {} done'.format(self.current_epoch)
            )

    def training_batch_start(self):
        self.training_start_timestamp = datetime.now()
        self.training_step += 1

    def training_batch_end(self, loss, train_true_energy, train_event_length):
        self.training_end_timestamp = datetime.now()
        self.train_time_delta.append(
            (
                self.training_end_timestamp - self.training_start_timestamp
            ).total_seconds()
        )
        self.train_loss.extend(loss)
        self.train_true_energy.extend(train_true_energy)
        self.train_event_length.extend(train_event_length)

    def val_batch_start(self):
        self.val_start_timestamp = datetime.now()
        self.val_step += 1

    def val_batch_end(self, loss):
        self.val_end_timestamp = datetime.now()
        self.val_time_delta.append(
            (
                self.val_end_timestamp - self.val_start_timestamp
            ).total_seconds()
        )
        self.train_loss.extend(loss)
        print(self.train_loss)

    def val_end(self, avg_train_loss, avg_val_loss):
        print('''
{}: Step {} / epoch {}
        Train loss: {:.3f} / {:.1f} events/s
        Val loss:   {:.3f} / {:.1f} events/s
                '''
            .format(
                get_time(),
                self.training_step,
                self.current_epoch,
                avg_train_loss,
                sum(self.train_step) * self.config.batch_size / sum(self.train_time_delta),
                avg_val_loss,
                sum(self.val_step) * self.config.val_batch_size / sum(self.val_time_delta)
            )
        )
        avg_train_loss = torch.stack(self.train_loss).mean()
        avg_val_loss = torch.stack(self.val_loss).mean()
        if self.config.wandb:
            metrics = {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}
            self.wandb.log(metrics, step=self.global_step)
        self.train_loss = []
        self.train_step = 0
        self.train_time_delta = []
        self.val_loss = []
        self.val_step = 0
        self.val_time_delta = []
        self.train_true_energy = []
        self.train_event_length = []