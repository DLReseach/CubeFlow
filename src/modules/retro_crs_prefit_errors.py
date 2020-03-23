import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


def unit_vector(vectors):
    '''Returns the unit vector of the vector.'''
    lengths = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    unit_vectors = vectors / lengths
    return unit_vectors


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


def angle_between(v1, v2):
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


def distance_between(p1, p2):
    '''Return the Euclidean distance between points 'p1' and 'p2'.

    Args:
        p1 (numpy.ndarray): point 1
        p2 (numpy.ndarray): point 2

    Returns:
        numpy.ndarray: distances between points 1 and 2
    '''
    distances = np.linalg.norm((p1 - p2), axis=1).reshape(-1, 1)
    return distances


def delta_energy_log(energy1, energy2):
    '''Return the difference in energy vectors in log space.

    Args:
        energy1 (numpy.ndarray): first energy vector
        energy2 (numpy.ndarray): second energy vector

    Returns:
        numpy.ndarray: difference between energy vectors 1 and 2 in logspace
    '''
    differences = np.log10(energy1 / energy2).reshape(-1, 1)
    return differences


def delta_time(time1, time2):
    '''Return the difference between two scalar values.

    Args:
        time1 (numpy.ndarray): first time vector
        time2 (numpy.ndarray): second time vector

    Returns:
        numpy.ndarray: difference between time vectors 1 and 2
    '''
    differences = (time1 - time2).reshape(-1, 1)
    return differences


test_set_db_path = Path().home().joinpath('Downloads/sqlite/test_set.db')

features = [
    'event',
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

con = sqlite3.connect(test_set_db_path)
query = 'SELECT {features} FROM scalar'.format(
    features=', '.join(features)
)
predictions = pd.read_sql_query(query, con)
predictions = predictions[predictions['retro_crs_prefit_energy'] > 0]

directions = convert_spherical_to_cartesian(
    predictions['retro_crs_prefit_zenith'].values,
    predictions['retro_crs_prefit_azimuth'].values
)

vertex_errors = distance_between(
    predictions[['retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z']].values,
    predictions[['true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z']].values
)

angle_errors = angle_between(
    directions,
    predictions[['true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']].values
)

time_errors = delta_time(
    predictions['retro_crs_prefit_time'].values,
    predictions['true_primary_time'].values
)

energy_errors = delta_energy_log(
    predictions['retro_crs_prefit_energy'].values,
    10**predictions['true_primary_energy'].values
)

out_df = pd.DataFrame(
    {
        'event': predictions['event'].values,
        'postition_x': predictions['retro_crs_prefit_x'].values,
        'postition_y': predictions['retro_crs_prefit_y'].values,
        'postition_z': predictions['retro_crs_prefit_z'].values,
        'azimuth': predictions['retro_crs_prefit_azimuth'].values,
        'zenith': predictions['retro_crs_prefit_zenith'].values,
        'direction_x': directions[:, 0],
        'direction_y': directions[:, 1],
        'direction_z': directions[:, 2],
        'time': predictions['retro_crs_prefit_time'].values,
        'energy': predictions['retro_crs_prefit_energy'].values,
        'vertex_error': vertex_errors[:, 0],
        'angle_error': angle_errors[:, 0],
        'time_error': time_errors[:, 0],
        'energy_error': energy_errors[:, 0]
    }
)

out_df.set_index('event', inplace=True)
out_df.to_sql('retro_crs_prefit', con=con, if_exists='replace')

con.close()
