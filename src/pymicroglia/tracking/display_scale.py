"""Exact positive-intensity display percentile with bounded movie scratch."""
import numpy as np


def _float_display_high(raw):
    """Exact positive order statistics, with one byte of the float key per pass.

    Positive IEEE float bit patterns have the same order as their values.
    Only two ranks are needed. Each pass retains two 256-bin histograms and
    frame blocks, rather than a movie-sized copy of all positive values.
    """
    dtype=raw.dtype.newbyteorder('=')
    unsigned=np.dtype('u'+str(dtype.itemsize))
    bits=dtype.itemsize*8
    targets=None
    for shift in range(bits-8,-1,-8):
        prefixes=[0] if targets is None else sorted({prefix for prefix,rank in targets})
        counts=np.zeros((len(prefixes),256),dtype=np.int64)
        for frame in raw:
            plain=frame.view(np.ndarray)
            flat=plain.reshape(-1) if plain.flags.c_contiguous else plain.flat
            for start in range(0,plain.size,65536):
                block=flat[start:start+65536]
                positive=block[block>0]
                keys=positive.astype(dtype,copy=False).view(unsigned)
                for row,prefix in enumerate(prefixes):
                    selected=keys if shift==bits-8 else keys[(keys>>(shift+8))==prefix]
                    bins=((selected>>shift)&255).astype(np.intp)
                    counts[row]+=np.bincount(bins,minlength=256)
        if targets is None:
            size=int(counts.sum())
            if not size:return 1.0
            quantile=np.asanyarray(np.true_divide(99.5,dtype.type(100)))
            position=np.asanyarray((size-1)*quantile)
            # NumPy may overflow its virtual rank for long float16 inputs.
            # Its infinite interpolation weight yields NaN in that case.
            if not np.isfinite(position):return float("nan")
            lower=int(np.floor(position))
            targets=[(0,lower),(0,min(lower+1,size-1))]
            fraction=np.asanyarray(position-np.intp(lower),dtype=position.dtype)
        cumulative=np.cumsum(counts,axis=1)
        next_targets=[]
        for prefix,rank in targets:
            row=prefixes.index(prefix)
            digit=int(np.searchsorted(cumulative[row],rank,side='right'))
            preceding=int(cumulative[row,digit-1]) if digit else 0
            next_targets.append(((prefix<<8)|digit,rank-preceding))
        targets=next_targets
    endpoints=np.array([prefix for prefix,rank in targets],dtype=unsigned).view(dtype)
    return float(np.quantile(endpoints,fraction,method='linear'))


def positive_display_high(raw):
    """Match percentile(raw[raw > 0], 99.5), including NumPy interpolation.

For 8/16-bit integer images, a bounded histogram identifies the same two
order statistics. Large floating movies use byte histograms to locate the
same endpoints. NumPy itself interpolates them. Small floating movies and
other dtypes retain the original percentile on private selected values.
"""
    if raw.dtype.kind in 'ui' and raw.dtype.itemsize <= 2:
        bins = 1 << (raw.dtype.itemsize * 8)
        counts = np.zeros(bins, dtype=np.int64)
        for frame in raw:
            values = frame.reshape(-1)
            # bincount converts integers to platform-width indices. Bound that
            # temporary too, including for very wide future camera frames.
            for start in range(0, values.size, 1024 * 1024):
                block = values[start:start + 1024 * 1024]
                if raw.dtype.kind == 'i': block = block[block > 0]
                counts += np.bincount(block, minlength=bins)
        counts[0] = 0
        n = int(counts.sum())
        if not n: return 1.0
        cumulative = np.cumsum(counts)
        position = (n - 1) * np.float64(99.5 / 100)
        lower = int(np.floor(position))
        upper = min(lower + 1, n - 1)
        endpoints = np.searchsorted(cumulative, [lower, upper], side='right').astype(raw.dtype)
        return float(np.quantile(endpoints, position - lower, method='linear'))
    if (raw.dtype.kind=='f' and raw.dtype.itemsize in (2,4,8) and raw.nbytes>32*1024**2
            and np.lib.NumpyVersion(np.__version__) >= '2.0.0'):
        return _float_display_high(raw)
    positive = raw[raw > 0]
    return float(np.percentile(positive, 99.5, overwrite_input=True)) if positive.size else 1.0
