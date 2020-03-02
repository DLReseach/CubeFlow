import torch
from datetime import datetime

from src.modules.utils import get_time


class Reporter:
    def __init__(self, config, wandb, client):
        super(Reporter, self).__init__()
        self.config = config
        self.wandb = wandb
        self.client = client

        self.current_epoch = 0
        self.train_loss = []
        self.training_step = 0
        self.val_loss = []
        self.val_step = 0
        self.train_true_energy = []
        self.train_event_length = []
        self.global_step = 0
        self.iteration = 0
        self.first_train = True
        self.first_val = True

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
        self.iteration = 0
        self.current_epoch += 1

    def training_batch_start(self):
        if self.first_train:
            self.training_start_timestamp = datetime.now()
            self.first_train = False
        self.training_step += 1
        self.global_step += 1

    def training_batch_end(self, loss):
        self.train_loss.append(loss)

    def val_batch_start(self):
        if self.first_val:
            self.training_end_timestamp = datetime.now()
            self.train_time_delta = (self.training_end_timestamp - self.training_start_timestamp).total_seconds()
            self.val_start_timestamp = datetime.now()
            self.first_val = False
        self.val_step += 1

    def val_batch_end(self, loss):
        self.val_loss.append(loss)

    def on_val_end(self):
        self.val_end_timestamp = datetime.now()
        self.val_time_delta = (self.val_end_timestamp - self.val_start_timestamp).total_seconds()
        self.iteration += 1
        avg_train_loss = torch.stack(self.train_loss).mean()
        avg_val_loss = torch.stack(self.val_loss).mean()
        log_text = ('''
{}: Step {} / epoch {}
        Train loss: {:.3f} / {:.1f} events/s
        Val loss:   {:.3f} / {:.1f} events/s
                '''
            .format(
                get_time(),
                self.iteration,
                self.current_epoch,
                avg_train_loss,
                self.training_step * self.config.batch_size / self.train_time_delta,
                avg_val_loss,
                self.val_step * self.config.val_batch_size / self.val_time_delta
            )
        )
        print(log_text)
        if not self.config.dev_run:
            self.client.chat_postMessage(
                channel='training',
                text=log_text
            )
        if self.config.wandb:
            metrics = {'train_loss': avg_train_loss, 'val_loss': avg_val_loss}
            self.wandb.log(metrics, step=self.global_step)
        self.train_loss = []
        self.training_step = 0
        self.val_loss = []
        self.val_step = 0
        self.train_true_energy = []
        self.train_event_length = []
        self.first_train = True
        self.first_val = True
        return avg_val_loss

    def optimizer_step(self, learning_rate):
        if self.config.wandb == True:
            self.wandb.log(
                {'learning_rate': learning_rate},
                step=self.global_step
            )

    def add_plot_to_wandb(self, im, log_text):
        self.wandb.log(
            {log_text: [self.wandb.Image(im)]}
        )

    def add_metric_comparison_to_wandb(self, markers, log_text):
        for x, y in zip(markers.get_data()[0], markers.get_data()[1]):
            self.wandb.log(
                {
                    log_text: y,
                    'energy': x
                }
            )

    def save_file_to_wandb(self, file_name):
        self.wandb.save(file_name)