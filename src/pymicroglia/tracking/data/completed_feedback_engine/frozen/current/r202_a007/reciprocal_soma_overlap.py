"""Connect each silhouette through reciprocal best normalized overlap.

Dice overlap normalizes the shared pixels by both observed silhouette areas.
Exact ties remain connected: arbitrary component order must not decide a cell.
"""
import numpy as np


def reciprocal_overlap_pairs(pairs, counts, areas):
    pairs=np.asarray(pairs,dtype=np.int64)
    if not len(pairs):return pairs.reshape((0,2))
    areas=np.asarray(areas)
    scores=2*np.asarray(counts)/(areas[pairs[:,0]]+areas[pairs[:,1]])
    best=np.zeros(len(areas),dtype=float)
    np.maximum.at(best,pairs[:,0],scores)
    np.maximum.at(best,pairs[:,1],scores)
    return pairs[(scores==best[pairs[:,0]]) & (scores==best[pairs[:,1]])]
