import numpy as np
def loops(labels, values):
    a = [np.any(labels == v, axis=(1, 2)) for v in values]
    b = [np.count_nonzero(labels == v, axis=(1, 2)) for v in values]
    c = [np.flatnonzero(np.any(labels == v, axis=(1, 2))) for v in values]
    d = [((labels == v) & (labels > 0)).sum() for v in values]
    return a, b, c, d
