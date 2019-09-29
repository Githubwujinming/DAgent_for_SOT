import numpy as np
import random
from collections import deque


class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0


    def sample(self, count):
        """
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        s_g = np.float32([arr[0] for arr in batch])
        s_g_ = np.float32([arr[1] for arr in batch])
        a_arr = np.float32([arr[2] for arr in batch])
        r_arr = np.float32([arr[3] for arr in batch])
        s_l = np.float32([arr[4] for arr in batch])
        s_l_ = np.float32([arr[5] for arr in batch])

        return s_g, s_g_, a_arr, r_arr,s_l, s_l_

    def len(self):
        return self.len

    def add(self, s_g, s_g_, a, r,s_l, s_l_):
        """
        adds a particular transaction in the memory buffer
        :param s: current state
        :param a: action taken
        :param r: reward received
        :param s1: next state
        :return:
        """
        transition = (s_g, s_g_, a, r,s_l, s_l_)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)