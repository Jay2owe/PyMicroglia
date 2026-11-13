"""Experimental physical image grid; not enabled by the tracking entry point.

All leading axes are retained. Equal calibrations return the original array.
Continuous pixels use linear interpolation; categorical identities use integer
indexing, so even large integer names are never rounded through floating point.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates
from detector_spatial_units import positive_length


@dataclass(frozen=True)
class PhysicalGrid:
    source_shape: tuple
    target_shape: tuple
    scale: float

    @classmethod
    def from_calibration(cls,shape,source_pixel_size_um,target_pixel_size_um,*,target_shape=None):
        if len(shape)!=2 or any(isinstance(v,bool) or int(v)!=v or v<=0 for v in shape):
            raise ValueError('A positive two-dimensional spatial shape is required')
        ratio=positive_length(source_pixel_size_um)/positive_length(target_pixel_size_um)
        spatial=tuple(int(v) for v in shape)
        desired=tuple(max(1,round(v*ratio)) for v in spatial) if target_shape is None else tuple(target_shape)
        if len(desired)!=2 or any(isinstance(v,bool) or int(v)!=v or v<=0 for v in desired):
            raise ValueError('A positive target shape is required')
        return cls(spatial,tuple(int(v) for v in desired),float(ratio))

    def transform_points(self,xy,*,inverse=False):
        values=np.asarray(xy,dtype=float)
        if values.shape[-1]!=2:raise ValueError('Points must have x/y pairs')
        return (values+.5)/self.scale-.5 if inverse else (values+.5)*self.scale-.5

    def apply(self,array,*,categorical=False):
        values=np.asarray(array)
        if values.ndim<2 or tuple(values.shape[-2:])!=self.source_shape:
            raise ValueError('Array spatial dimensions do not match the grid')
        if self.scale==1.0 and self.target_shape==self.source_shape:return values
        ys=(np.arange(self.target_shape[0],dtype=float)+.5)/self.scale-.5
        xs=(np.arange(self.target_shape[1],dtype=float)+.5)/self.scale-.5
        if categorical:
            yi=np.clip(np.floor(ys+.5).astype(np.int64),0,self.source_shape[0]-1)
            xi=np.clip(np.floor(xs+.5).astype(np.int64),0,self.source_shape[1]-1)
            return values[...,yi[:,None],xi[None,:]]
        yy,xx=np.meshgrid(ys,xs,indexing='ij')
        result=np.empty((*values.shape[:-2],*self.target_shape),np.float32)
        for index in np.ndindex(values.shape[:-2]):
            result[index]=map_coordinates(values[index].astype(np.float32,copy=False),[yy,xx],order=1,mode='nearest',prefilter=False)
        return result
