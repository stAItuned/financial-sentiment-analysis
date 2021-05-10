import numpy as np


def scale_range(input, min, max):
    input = np.array(input)
    input += -(np.min(input))
    input = input / np.max(input) / (max - min)
    input += min

    return input.tolist()