import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


class ErrorCalculator():
    def __init__(self, run_name, first_run, dirs):
        self.run_name = run_name
        self.first_run = first_run
        self.dirs = dirs

        self.predictions_db_path = dirs['dbs'].joinpath('predictions.db')
        self.errors_db_path = dirs['dbs'].joinpath('errors.db')

        pd.options.mode.use_inf_as_na = True

        if first_run:
            self.calculate_errors('retro_crs_prefit')
        self.calculate_errors(run_name)

    def convert_spherical_to_cartesian(self, zenith, azimuth):
        '''Convert spherical coordinates to Cartesian coordinates.

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


    def angle_between(self, v1, v2):
        '''Returns the angle in radians between vectors 'v1' and 'v2'.
        
        Accounts for opposite vectors using numpy.clip.

        Args:
            v1 (numpy.ndarray): vector 1
            v2 (numpy.ndarray): vector 2

        Returns:
            numpy.ndarray: angles between vectors 1 and 2
        '''
        p1 = np.einsum('ij,ij->i', v1, v2)
        p2 = np.einsum('ij,ij->i', v1, v1)
        p3 = np.einsum('ij,ij->i', v2, v2)
        p4 = p1 / np.sqrt(p2 * p3)
        angles = np.arccos(np.clip(p4, -1.0, 1.0)).reshape(-1, 1)
        return angles


    def distance_between(self, p1, p2):
        '''Return the Euclidean distance between points 'p1' and 'p2'.

        Args:
            p1 (numpy.ndarray): point 1
            p2 (numpy.ndarray): point 2

        Returns:
            numpy.ndarray: distances between points 1 and 2
        '''
        distances = np.linalg.norm((p1 - p2), axis=1).reshape(-1, 1)
        return distances


    def delta_energy_log(self, energy1, energy2):
        '''Return the difference in energy vectors in log space.
        
        Assumes both energy vectors are log(energy).
        The calculation is thus equal to log(energy1 / energy2) if the
        energy vectors are not log.

        Args:
            energy1 (numpy.ndarray): first energy vector
            energy2 (numpy.ndarray): second energy vector

        Returns:
            numpy.ndarray: difference between energy vectors 1 and 2 in logspace
        '''
        differences = (energy1 - energy2).reshape(-1, 1)
        return differences


    def simple_delta(self, time1, time2):
        '''Return the difference between two scalar values.

        Args:
            time1 (numpy.ndarray): first time vector
            time2 (numpy.ndarray): second time vector

        Returns:
            numpy.ndarray: difference between time vectors 1 and 2
        '''
        differences = (time1 - time2).reshape(-1, 1)
        return differences

    def calculate_errors(self, run_name):
        predictions_query = 'SELECT * FROM \"{table}\"'.format(
            table=run_name
        )
        truths_query = 'SELECT * FROM truth'

        with sqlite3.connect(self.predictions_db_path) as con:
            predictions = pd.read_sql_query(predictions_query, con=con)
            truths = pd.read_sql_query(truths_query, con=con)

        vertex_errors = self.distance_between(
            predictions[['position_x', 'position_y', 'position_z']].values,
            truths[['position_x', 'position_y', 'position_z']].values
        )

        angle_errors = self.angle_between(
            predictions[['direction_x', 'direction_y', 'direction_z']].values,
            truths[['direction_x', 'direction_y', 'direction_z']].values
        )
        energy_errors = self.delta_energy_log(
            predictions['energy'].values,
            truths['energy'].values
        )
        error_dictionary = {
            'event_no': predictions['event_no'].values,
            'vertex': vertex_errors[:, 0],
            'angle': angle_errors[:, 0],
            'energy': energy_errors[:, 0]
        }
        for target in predictions.columns.values:
            if target == 'event_no' or target == 'energy':
                pass
            else:
                error_dictionary[target] = self.simple_delta(
                    predictions[target].values,
                    truths[target].values
                ).flatten()
        errors_to_sql = pd.DataFrame(
            error_dictionary
        )

        errors_to_sql.set_index('event_no', inplace=True)

        with sqlite3.connect(self.errors_db_path) as con:
            errors_to_sql.to_sql(run_name, con=con, if_exists='replace')
