import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


def convert_spherical_to_cartesian(zenith, azimuth):
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


test_set_db_path = Path().home().joinpath('CubeFlowData').joinpath('db').joinpath('test_set.db')
predictions_db_path = Path().home().joinpath('CubeFlowData').joinpath('db').joinpath('predictions.db')
errors_db_path = Path().home().joinpath('CubeFlowData').joinpath('db').joinpath('errors.db')

features = [
    'event_no',
    'retro_crs_prefit_x',
    'retro_crs_prefit_y',
    'retro_crs_prefit_z',
    'retro_crs_prefit_azimuth',
    'retro_crs_prefit_zenith',
    'retro_crs_prefit_time',
    'retro_crs_prefit_energy',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z',
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
    'true_primary_time',
    'true_primary_energy'
]

query = 'SELECT {features} FROM scalar'.format(
    features=', '.join(features)
)

with sqlite3.connect(test_set_db_path) as con:
    predictions = pd.read_sql_query(query, con)

directions = convert_spherical_to_cartesian(
    predictions['retro_crs_prefit_zenith'].values,
    predictions['retro_crs_prefit_azimuth'].values
)

predictions_to_sql = pd.DataFrame(
    {
        'event': predictions['event_no'].values,
        'position_x': predictions['retro_crs_prefit_x'].values,
        'position_y': predictions['retro_crs_prefit_y'].values,
        'position_z': predictions['retro_crs_prefit_z'].values,
        'direction_x': directions[:, 0],
        'direction_y': directions[:, 1],
        'direction_z': directions[:, 2],
        'time': predictions['retro_crs_prefit_time'].values,
        'energy': predictions['retro_crs_prefit_energy'].values,
    }
)

predictions_to_sql.set_index('event', inplace=True)

with sqlite3.connect(predictions_db_path) as con:
    predictions_to_sql.to_sql('retro_crs_prefit', con=con, if_exists='replace')
