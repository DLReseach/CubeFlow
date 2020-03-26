import torch
import numpy as np
import pandas as pd
import sqlite3

from src.modules.utils import get_time
from src.modules.transform_inverter import InvertTransforms

class Inferer:
    def __init__(
        self,
        model,
        optimizer,
        loss,
        dataset,
        config,
        experiment_name,
        files_and_dirs
    ):
        super(Inferer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.config = config
        self.experiment_name = experiment_name
        self.files_and_dirs = files_and_dirs

        self.transform_inverter = InvertTransforms(files_and_dirs['transformers'].joinpath('sqlite_transformers.pickle'))
        self.sql_file = files_and_dirs['dbs'].joinpath('predictions.db')

        gpu = self.config['gpulab_gpus'] if self.config['gpulab'] else self.config['gpus']
        self.device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')        
        self.model.to(self.device)

        self.data = {'event_no': []}
        self.predictions = {target: [] for target in self.config['targets']}

    def infer(self, model_path, save_path):
        loss = []
        self.create_dataloader()
        if not torch.cuda.is_available():
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i % 10 == 0 and i > 0:
                    print('{}: Inferred on {} events'.format(get_time(), i * self.config['val_batch_size']))
                x = batch[0].to(self.device).float()
                y = batch[1].to(self.device).float()
                events = batch[2].to(self.device)
                y_hat = self.model.forward(x)
                loss.append(self.loss(y_hat, y))
                self.on_test_step(y_hat, events)
                # if i == 12:
                #     break
            self.on_test_end()

    def on_test_step(self, y_hat, events):
        self.data['event_no'].extend(events.numpy())
        for i, target in enumerate(self.config['targets']):
            self.predictions[target].extend(y_hat[:, i].numpy())

    def on_test_end(self):
        for key in self.predictions:
            self.predictions[key] = np.array(self.predictions[key])
        transformed_y_hat = self.transform_inverter.invert_transform(self.predictions)
        self.data.update(transformed_y_hat)
        self.save_to_db()

    def save_to_db(self):
        predictions = pd.DataFrame().from_dict(self.data)
        predictions.set_index('event_no', inplace=True)
        predictions.rename(
            columns={
                key: key.replace('true_primary_', '') for key in predictions.columns
            },
            inplace=True
        )
        if all(x in predictions.columns.values for x in ['direction_x', 'direction_y', 'direction_z']):
            predictions['azimuth'], predictions['zenith'] = self.convert_cartesian_to_spherical(
                predictions[['direction_x', 'direction_y', 'direction_z']].values
            )
        predictions.sort_values(by='event_no', inplace=True)
        with sqlite3.connect(self.sql_file) as con:
            predictions.to_sql(self.experiment_name, con=con, if_exists='replace')

    def convert_cartesian_to_spherical(self, vectors):
        '''Convert Cartesian coordinates to signed spherical coordinates.
        
        Converts Cartesian vectors to unit length before conversion.

        Args:
            vectors (numpy.ndarray): x, y, z coordinates in shape (n, 3)

        Returns:
            tuple: tuple containing:
                azimuth (numpy.ndarray): signed azimuthal angles
                zenith (numpy.ndarray): zenith/polar angles
        '''
        lengths = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
        unit_vectors = vectors / lengths
        x = unit_vectors[:, 0]
        y = unit_vectors[:, 1]
        z = unit_vectors[:, 2]
        azimuth = np.arctan2(y, x).reshape(-1, 1)
        zenith = np.arccos(z).reshape(-1, 1)
        return azimuth, zenith

    def create_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=4,
            shuffle=False
        )