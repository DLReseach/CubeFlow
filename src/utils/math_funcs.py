import numpy as np


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(vector_1, vector_2):
    vector_1_unit = unit_vector(vector_1)
    vector_2_unit = unit_vector(vector_2)
    return np.arccos(np.clip(np.dot(vector_1_unit, vector_2_unit), -1.0, 1.0))
