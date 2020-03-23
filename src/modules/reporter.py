import torch
from datetime import datetime

from src.modules.utils import get_time


class Reporter:
    def __init__(self, config, wandb, client, experiment_name):
        super(Reporter, self).__init__()
        self.config = config
        self.wandb = wandb
        self.client = client
        self.experiment_name = experiment_name

        self.current_epoch = 0
        self.train_loss = []
        self.training_step = 0
        self.val_loss = []
        self.val_step = 0
        self.global_step = 0
        self.iteration = 0
        self.first_train = True
        self.first_val = True

    def on_epoch_start(self):
        print(
            '''
{}: {}: beginning epoch {}
            '''
            .format(get_time(), self.experiment_name, self.current_epoch)
        )
        if not self.config.dev_run and self.client is not None:
            self.client.chat_postMessage(
                channel='training',
                text='{}: beginning epoch {}'.format(self.experiment_name, self.current_epoch)
            )

    def on_epoch_end(self):
        self.iteration = 0
        self.current_epoch += 1

    def on_training_batch_start(self):
        if self.first_train:
            self.training_start_timestamp = datetime.now()
            self.first_train = False
        self.training_step += 1
        self.global_step += 1

    def on_training_batch_end(self, loss):
        self.train_loss.append(loss)

    def on_intermediate_training_end(self):
        self.avg_train_loss = torch.stack(self.train_loss).mean()
        self.training_end_timestamp = datetime.now()
        self.train_time_delta = (self.training_end_timestamp - self.training_start_timestamp).total_seconds()
        log_text = ('''
{}: Iteration {} / epoch {}
        Train loss: {:.3f} / {} batches / {:.1f} events/s
                '''
            .format(
                get_time(),
                self.iteration,
                self.current_epoch,
                self.avg_train_loss,
                len(self.train_loss),
                len(self.train_loss) * self.config.batch_size / self.train_time_delta,
            )
        )
        print(log_text)
        if not self.config.dev_run and self.client is not None:
            self.client.chat_postMessage(
                channel='training',
                text=log_text
            )

    def on_intermediate_validation_batch_start(self):
        if self.first_val:
            self.val_start_timestamp = datetime.now()
            self.first_val = False
        self.val_step += 1

    def on_intermediate_validation_batch_end(self, loss):
        self.val_loss.append(loss)

    def on_intermediate_validation_end(self):
        self.val_end_timestamp = datetime.now()
        self.val_time_delta = (self.val_end_timestamp - self.val_start_timestamp).total_seconds()
        avg_val_loss = torch.stack(self.val_loss).mean()
        log_text = ('''
{}: Iteration {} / epoch {}
        Val loss:   {:.3f} / {} batches / {:.1f} events/s
                '''
            .format(
                get_time(),
                self.iteration,
                self.current_epoch,
                avg_val_loss,
                len(self.val_loss),
                len(self.val_loss) * self.config.val_batch_size / self.val_time_delta
            )
        )
        print(log_text)
        if not self.config.dev_run and self.client is not None:
            self.client.chat_postMessage(
                channel='training',
                text=log_text
            )
        if self.config.wandb:
            metrics = {'train_loss': self.avg_train_loss, 'val_loss': avg_val_loss}
            self.wandb.log(metrics, step=self.global_step)
        self.iteration += 1
        self.train_loss = []
        self.training_step = 0
        self.val_loss = []
        self.val_step = 0
        self.first_train = True
        self.first_val = True
        self.avg_train_loss = 0

    def on_epoch_validation_batch_start(self):
        if self.first_val:
            self.val_start_timestamp = datetime.now()
            self.first_val = False
        self.val_step += 1

    def on_epoch_validation_batch_end(self, loss):
        self.val_loss.append(loss)

    def on_epoch_validation_end(self):
        self.val_end_timestamp = datetime.now()
        self.val_time_delta = (self.val_end_timestamp - self.val_start_timestamp).total_seconds()
        self.iteration += 1
        avg_val_loss = torch.stack(self.val_loss).mean()
        log_text = ('''
{}: Epoch validation / epoch {}
        Val loss:   {:.3f} / {} batches / {:.1f} events/s
                '''
            .format(
                get_time(),
                self.current_epoch,
                avg_val_loss,
                len(self.val_loss),
                len(self.val_loss) * self.config.val_batch_size / self.val_time_delta
            )
        )
        print(log_text)
        if not self.config.dev_run and self.client is not None:
            self.client.chat_postMessage(
                channel='training',
                text=log_text
            )
        self.train_loss = []
        self.training_step = 0
        self.val_loss = []
        self.val_step = 0
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