import numpy as np


def cal_distance(samples, ground_th):
    distance = samples[:, 0:2] + samples[:, 2:4] / 2.0 - ground_th[:, 0:2] - ground_th[:, 2:4] / 2.0
    distance = distance / samples[:, 2:4]
    rate = ground_th[:, 3] / samples[:, 3]
    rate = np.array(rate).reshape(rate.shape[0], 1)
    rate = rate - 1.0
    distance = np.hstack([distance, rate])

    return distance
