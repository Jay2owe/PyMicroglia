"""Compare a resolved soma's radial intensity curvature with a local slope.

Coordinates and intensity are normalised to the measured enclosure. The model
comparison uses the Gaussian residual information criterion n*log(SSE/n)+k*log(n),
with one extra curvature coefficient; there is no fitted acceptance cutoff.
Unresolved designs remain ambiguous and cannot be evidence for removal.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image


def soma_profile_supported(body, patch):
    enclosure=convex_hull_image(body)
    if not ndi.binary_erosion(enclosure,ndi.generate_binary_structure(2,1)).any():
        return True
    y,x=np.nonzero(enclosure)
    if len(x)<=4:return True
    x=x.astype(float)-x.mean();y=y.astype(float)-y.mean()
    scale=max(np.ptp(x),np.ptp(y),1.0);x/=scale;y/=scale
    values=patch[enclosure].astype(float)
    values-=values.mean();amplitude=np.max(np.abs(values))
    if not np.isfinite(values).all() or amplitude==0:return True
    values/=amplitude
    plane=np.column_stack((np.ones(len(x)),x,y))
    dome=np.column_stack((plane,x*x+y*y))
    plane_fit,_,plane_rank,_=np.linalg.lstsq(plane,values,rcond=None)
    dome_fit,_,dome_rank,_=np.linalg.lstsq(dome,values,rcond=None)
    if plane_rank<3 or dome_rank<4:return True
    if dome_fit[-1]>=0:return False
    plane_error=np.sum((values-plane@plane_fit)**2)
    dome_error=np.sum((values-dome@dome_fit)**2)
    floor=np.finfo(float).eps**2*len(values)
    gain=len(values)*np.log(max(plane_error,floor)/max(dome_error,floor))-np.log(len(values))
    return bool(gain>0)
