import numpy as np


def most_frequent_in_rows(a):
    out = np.empty((a.shape[0], 1))
    for i, row in enumerate(a):
        out[i, 0] = np.bincount(row.astype(int)).argmax()
    return out
