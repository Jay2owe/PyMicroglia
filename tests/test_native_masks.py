import numpy as np
import tifffile
from pymicroglia.tracking.native_masks import project

def test_native_projection_keeps_clock_and_identity_ownership(tmp_path):
    shape=(8,11);poses=np.repeat(np.eye(3)[None],4,axis=0);poses[:,0,2]=[0,1,2,3]
    np.save(tmp_path/'poses.npy',poses)
    labels=np.zeros((2,8,16),np.uint16)
    for t in range(2):labels[t,2:4,3+t+2:5+t+2]=9;labels[t,5:7,8+t+2:10+t+2]=31
    path=tmp_path/'labels.tif';tifffile.imwrite(path,labels,photometric='minisblack',metadata={'axes':'TYX'})
    out=project({'labels':dict(path=path)},tmp_path/'poses.npy',native_shape=shape,output_dir=tmp_path/'native',source_frame_offset=2)
    native=tifffile.imread(out['outputs']['labels']['path'])
    for frame in native:assert np.all(frame[2:4,3:5]==9);assert np.all(frame[5:7,8:10]==31)
    renamed=np.where(labels==9,301,np.where(labels==31,17,0)).astype(np.uint16)
    tifffile.imwrite(tmp_path/'renamed.tif',renamed,photometric='minisblack')
    alt=project({'labels':dict(path=tmp_path/'renamed.tif')},tmp_path/'poses.npy',native_shape=shape,output_dir=tmp_path/'alt',source_frame_offset=2)
    np.testing.assert_array_equal(tifffile.imread(alt['outputs']['labels']['path']),np.where(native==9,301,np.where(native==31,17,0)))
