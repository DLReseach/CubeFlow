from datetime import datetime
import pandas as pd

from src.modules.utils import get_time


class Saver:
    def __init__(self, config, files_and_dirs):
        super(Saver, self).__init__()
        self.config = config
        self.files_and_dirs = files_and_dirs

        self.data = {}

    def update(
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

    def val_end(self):
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
            train_dists_file_name = self.files_and_dirs['run_root'].joinpath(
                'train_dists_parquet.gzip'
            )
            train_dists_df.to_parquet(
                str(train_dists_file_name),
                compression='gzip'
            )

    def reset(self):
        self.data = {}
