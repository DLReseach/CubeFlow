import torch
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import pickle

from src.modules.utils import get_time
from src.modules.utils import get_project_root


class Saver:
    def __init__(self, config, files_and_dirs):
        super(Saver, self).__init__()
        self.config = config
        self.files_and_dirs = files_and_dirs

        self.train_true_energy = []
        self.train_event_length = []
        self.epoch_val_loss = []
        self.early_stopping_counter = 0

        self.epoch = 0

        train_loss_to_pickle = []
        self.train_loss_pickle_file = files_and_dirs['run'].joinpath('train_loss.pkl')
        with open(self.train_loss_pickle_file, 'wb') as f:
            pickle.dump(train_loss_to_pickle, f)

        val_loss_to_pickle = []
        self.val_loss_pickle_file = files_and_dirs['run'].joinpath('val_loss.pkl')
        with open(self.val_loss_pickle_file, 'wb') as f:
            pickle.dump(val_loss_to_pickle, f)

        config_file = get_project_root().joinpath('configs').joinpath('config.json')
        shutil.copy(config_file, self.files_and_dirs['run'].joinpath('config.json'))

    def save_loss(self, train_loss, val_loss):
        with open(self.train_loss_pickle_file, 'rb') as f:
            train_loss_to_pickle = pickle.load(f)
        train_loss_to_pickle.append(train_loss.item())
        with open(self.train_loss_pickle_file, 'wb') as f: 
            pickle.dump(train_loss_to_pickle, f)
        with open(self.val_loss_pickle_file, 'rb') as f:
            val_loss_to_pickle = pickle.load(f)
        val_loss_to_pickle.append(val_loss.item())
        with open(self.val_loss_pickle_file, 'wb') as f:
            pickle.dump(val_loss_to_pickle, f)

    def early_stopping(self, epoch, epoch_val_loss, model_state_dict, optimizer_state_dict):
        epoch_val_loss = round(epoch_val_loss.item(), 3)
        if epoch == 0 or epoch_val_loss < min(self.epoch_val_loss):
            best_val_loss = epoch_val_loss
            self.save_model_state(epoch, model_state_dict, optimizer_state_dict)
            self.early_stopping_counter = 0
            print('{}: best model yet, saving'.format(get_time()))
        else:
            self.early_stopping_counter += 1
            print('{}: model didn\'t improve for {} epoch(s)'.format(get_time(), self.early_stopping_counter))
        self.epoch_val_loss.append(epoch_val_loss)
        if self.early_stopping_counter >= self.config['patience']:
            return True
        else:
            return False
    
    def save_model_state(self, epoch, model_state_dict, optimizer_state_dict):
        model_path = self.files_and_dirs['run'].joinpath('model.pt')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict
            },
            model_path
        )
