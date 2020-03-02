from datetime import datetime
import pandas as pd

from src.modules.utils import get_time
from src.modules.invert_transforms import TransformsInverter


class Saver:
    def __init__(self, config, wandb, files_and_dirs):
        super(Saver, self).__init__()
        self.config = config
        self.wandb = wandb
        self.files_and_dirs = files_and_dirs

        self.train_true_energy = []
        self.train_event_length = []

        self.transform_object = TransformsInverter(self.config, self.files_and_dirs)

        self.column_names = [
            'file_number',
            'energy',
            'event_length'
        ]
        self.column_names += ['opponent_' + name for name in self.config.comparison_metrics]
        self.column_names += ['own_' + name.replace('true_', '') for name in self.config.targets]
        self.column_names += [name for name in self.config.targets]
        self.data = {name: [] for name in self.column_names}

        self.first_run = True

    def train_step(self, train_true_energy, train_event_length):
        if self.config.save_train_dists:
            self.train_true_energy.extend(train_true_energy.tolist())
            self.train_event_length.extend(train_event_length.tolist())

    def on_val_step(
        self,
        x,
        y,
        y_hat,
        comparisons,
        energy,
        event_length,
        file_number
    ):
        values = [
            list(file_number),
            energy.tolist(),
            event_length.tolist(),
            *[comparison.tolist() for comparison in comparisons],
            *[y_hat[:, i].tolist() for i in range(y_hat.size(1))],
            *[y[:, i].tolist() for i in range(y.size(1))]
        ]
        for i, key in enumerate(self.data):
            self.data[key].extend(values[i])

    def on_val_end(self):
        if self.first_run:
            self.first_run = False
        else:
            self.data = self.transform_object.transform_inversion(self.data)
            comparison_df = pd.DataFrame().from_dict(self.data)
            file_name = self.files_and_dirs['run_root'].joinpath(
                'comparison_dataframe_parquet.gzip'
            )
            comparison_df.to_parquet(
                str(file_name),
                compression='gzip'
            )
            if self.config.save_train_dists:
                train_dists_dict = {}
                train_dists_dict['train_true_energy'] = self.train_true_energy
                train_dists_dict['train_event_length'] = self.train_event_length
                train_dists_df = pd.DataFrame().from_dict(train_dists_dict)
                train_dists_file_name = self.files_and_dirs['train_dists_path'].joinpath(
                    'train_dists_parquet.gzip'
                )
                train_dists_df.to_parquet(
                    str(train_dists_file_name),
                    compression='gzip'
                )
            self.data = {name: [] for name in self.column_names}
