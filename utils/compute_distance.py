import numpy as np


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2))
    b = np.array((boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist

