import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


def convert_spherical_to_cartesian(zenith, azimuth):
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


def convert_cartesian_to_spherical(vectors):
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
con = sqlite3.connect(test_set_db_path)

run_name = 'pompous-puma'
run_dataframe = Path().home().joinpath('Downloads/runs').joinpath(run_name).joinpath('prediction_dataframe_parquet.gzip')

predictions = pd.read_parquet(str(run_dataframe), engine='fastparquet')

azimuth, zenith = convert_cartesian_to_spherical(
    predictions[['own_primary_direction_x', 'own_primary_direction_y', 'own_primary_direction_z']].values
)

vertex_errors = distance_between(
    predictions[['own_primary_position_x', 'own_primary_position_y', 'own_primary_position_z']].values,
    predictions[['true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z']].values
)

angle_errors = angle_between(
    predictions[['own_primary_direction_x', 'own_primary_direction_y', 'own_primary_direction_z']].values,
    predictions[['true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']].values
)

time_errors = delta_time(
    predictions['own_primary_time'].values,
    predictions['true_primary_time'].values
)

energy_errors = delta_energy_log(
    10**predictions['own_primary_energy'].values,
    10**predictions['true_primary_energy'].values
)

out_df = pd.DataFrame(
    {
        'event': predictions['file_number'].values,
        'postition_x': predictions['own_primary_position_x'].values,
        'postition_y': predictions['own_primary_position_y'].values,
        'postition_z': predictions['own_primary_position_z'].values,
        'azimuth': azimuth[:, 0],
        'zenith': zenith[:, 0],
        'direction_x': predictions['own_primary_direction_x'].values,
        'direction_y': predictions['own_primary_direction_y'].values,
        'direction_z': predictions['own_primary_direction_z'].values,
        'time': predictions['own_primary_time'].values,
        'energy': 10**predictions['own_primary_energy'].values,
        'vertex_error': vertex_errors[:, 0],
        'angle_error': angle_errors[:, 0],
        'time_error': time_errors[:, 0],
        'energy_error': energy_errors[:, 0]
    }
)

out_df.set_index('event', inplace=True)
out_df.to_sql(run_name, con=con, if_exists='replace')

con.close()
