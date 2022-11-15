
from scipy.sparse import lil_matrix
import numpy as np
def sparse_matrix(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 7 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1

    return A