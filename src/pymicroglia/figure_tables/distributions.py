"""Prepare exact histogram bands before a renderer receives them."""
import numpy as np
import pandas as pd


def histogram(values,bins=24,*,log=False):
    values = np.asarray(values,dtype=float)
    values = values[np.isfinite(values)]
    if isinstance(bins,(int,np.integer)):
        if bins<1:
            raise ValueError('bins must be positive')
        if log:
            positive = values[values>0]
            low,high = (float(positive.min()),float(positive.max())) if positive.size else (1e-3,1.)
            if high<=low:low,high=low/np.sqrt(10),high*np.sqrt(10)
            edges = np.logspace(np.log10(low),np.log10(high),int(bins)+1)
            edges[0],edges[-1] = low,high
        else:
            low,high=(float(values.min()),float(values.max())) if values.size else (0.,1.)
            if high<=low:low,high=low-.5,high+.5
            edges = np.linspace(low,high,int(bins)+1)
    else:
        edges = np.asarray(bins,dtype=float)
    counts,_ = np.histogram(values,bins=edges)
    return pd.DataFrame({'bin_left':edges[:-1],'bin_right':edges[1:],'count':counts})
