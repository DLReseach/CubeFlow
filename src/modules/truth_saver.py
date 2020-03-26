import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pickle


class TruthSaver:
    def __init__(self, config, files_and_dirs, events):
        self.config = config
        self.files_and_dirs = files_and_dirs
        self.events = events
        self.targets = ['event_no'] + config['targets']

        self.test_set_db_path = Path().home().joinpath('CubeFlowData').joinpath('dbs').joinpath('test_set.db')
        self.predictions_db_path = files_and_dirs['dbs'].joinpath('predictions.db')

        self.save_truths_to_db()
        self.save_retro_crs_prefit_to_db()

    def save_truths_to_db(self):
        query = 'SELECT {targets} FROM scalar'.format(
            targets=', '.join(self.targets)
        )
        with sqlite3.connect(self.test_set_db_path) as con:
            truths = pd.read_sql_query(query, con)

        truths = truths[truths['event_no'].isin(self.events)]
        truths.rename(
            columns={
                key: key.replace('true_primary_', '') for key in truths.columns
            },
            inplace=True
        )
        if 'direction_x' in truths.columns.values and 'direction_y' in truths.columns.values and 'direction_z' in truths.columns.values:
            truths['azimuth'], truths['zenith'] = self.convert_cartesian_to_spherical(
                truths[['direction_x', 'direction_y', 'direction_z']].values
            )
        truths.set_index('event_no', inplace=True)
        truths.sort_values(by='event_no', inplace=True)
        with sqlite3.connect(self.predictions_db_path) as con:
            truths.to_sql('truth', con=con, if_exists='replace')

    def save_retro_crs_prefit_to_db(self):
        angle_prediction = False
        retro_crs_prefit_targets = []
        temp_targets = [target.replace('true_primary_', 'retro_crs_prefit_') for target in self.targets]
        for entry in temp_targets:
            if 'direction' in entry:
                angle_prediction = True
            elif 'position' in entry:
                retro_crs_prefit_targets.append(entry.replace('position_', ''))
            else:
                retro_crs_prefit_targets.append(entry)
        if angle_prediction:
            retro_crs_prefit_targets += ['retro_crs_prefit_azimuth', 'retro_crs_prefit_zenith']
        query = 'SELECT {targets} FROM scalar'.format(
            targets=', '.join(retro_crs_prefit_targets)
        )
        with sqlite3.connect(self.test_set_db_path) as con:
            retro_crs_prefit = pd.read_sql_query(query, con)
        retro_crs_prefit = retro_crs_prefit[retro_crs_prefit['event_no'].isin(self.events)]
        if 'retro_crs_prefit_energy' in retro_crs_prefit.columns.values:
            retro_crs_prefit['retro_crs_prefit_energy'] = np.log10(retro_crs_prefit['retro_crs_prefit_energy'].values)
        if angle_prediction:
            retro_crs_prefit['retro_crs_prefit_azimuth'] = self.convert_to_signed_angle(
                retro_crs_prefit['retro_crs_prefit_azimuth'].values, 'azimuth'
            )
            retro_crs_prefit['retro_crs_prefit_zenith'] = self.convert_to_signed_angle(
                retro_crs_prefit['retro_crs_prefit_zenith'].values, 'zenith'
            )
            directions = self.convert_spherical_to_cartesian(
                retro_crs_prefit['retro_crs_prefit_zenith'].values,
                retro_crs_prefit['retro_crs_prefit_azimuth'].values
            )
            retro_crs_prefit['retro_crs_prefit_direction_x'] = directions[:, 0]
            retro_crs_prefit['retro_crs_prefit_direction_y'] = directions[:, 1]
            retro_crs_prefit['retro_crs_prefit_direction_z'] = directions[:, 2]
        if 'retro_crs_prefit_x' in retro_crs_prefit.columns.values:
            retro_crs_prefit.rename(
                columns={
                    'retro_crs_prefit_x': 'retro_crs_prefit_position_x'
                },
                inplace=True
            )
        if 'retro_crs_prefit_y' in retro_crs_prefit.columns.values:
            retro_crs_prefit.rename(
                columns={
                    'retro_crs_prefit_y': 'retro_crs_prefit_position_y'
                },
                inplace=True
            )
        if 'retro_crs_prefit_z' in retro_crs_prefit.columns.values:
            retro_crs_prefit.rename(
                columns={
                    'retro_crs_prefit_z': 'retro_crs_prefit_position_z'
                },
                inplace=True
            )
        retro_crs_prefit.rename(
            columns={
                key: key.replace('retro_crs_prefit_', '') for key in retro_crs_prefit.columns
            },
            inplace=True
        )
        retro_crs_prefit.set_index('event_no', inplace=True)
        retro_crs_prefit.sort_values(by='event_no', inplace=True)
        with sqlite3.connect(self.predictions_db_path) as con:
            retro_crs_prefit.to_sql('retro_crs_prefit', con=con, if_exists='replace')

    def convert_spherical_to_cartesian(self, zenith, azimuth):
        '''Convert spherical coordinates to cartesian.

        Assumes unit length.

        Zenith: theta
        Azimuth: phi

        Args:
            zenith (numpy.ndarray): zenith/polar angle
            azimuth (numpy.ndarray): azimuthal angle

        Returns:
            numpy.ndarray: x, y, z (event, coordinates) vector
        '''
        theta = zenith
        phi = azimuth
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        vectors = np.array((x, y, z)).T
        return vectors

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

    def convert_to_signed_angle(self, angles, angle_type):
        if angle_type == 'azimuth':
            signed_angles = [
                entry if entry < np.pi else entry - 2 * np.pi for entry in angles
            ]
            reversed_angles = (
                [entry - np.pi if entry > 0 else entry + np.pi for entry in signed_angles]
            )
        elif angle_type == 'zenith':
            reversed_angles = np.pi - angles
        else:
            print('Unknown angle type')
        return reversed_angles
