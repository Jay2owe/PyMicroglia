"""Project tracked identities onto untouched original camera pixel centres."""
from pathlib import Path
import json
import numpy as np
import tifffile
from auto_organotypic.rigid_registration import to_native
from .prepare import sha256

METHOD_VERSION='2026-10-09-native-camera-mask-projection-v1'

def project(inputs,transforms,*,native_shape,output_dir,source_frame_offset=0):
    """Stream named label stacks through a shared source-to-canvas pose table.

    Each input is a mapping with path and source_frame_offset. The top-level
    offset supplies the default for tracked outputs; full detector foreground
    explicitly uses zero. No intensity image enters this operation.
    """
    matrices=np.load(transforms);shape=tuple(map(int,native_shape))
    if matrices.ndim!=3 or matrices.shape[1:]!=(3,3) or not np.isfinite(matrices).all():raise ValueError('Finite source-to-canvas poses are required')
    target=Path(output_dir);target.mkdir(parents=True,exist_ok=True);records={}
    for name,spec in inputs.items():
        path=Path(spec['path']);offset=int(spec.get('source_frame_offset',source_frame_offset))
        if offset<0:raise ValueError('Native mask source offset cannot be negative')
        output=target/f'{name}.tif'
        if output.exists():raise FileExistsError('Preserve previously projected masks; choose a fresh folder')
        with tifffile.TiffFile(path) as opened:
            count=len(opened.pages);dtype=opened.series[0].dtype
            if count+offset!=len(matrices):raise ValueError('Label frames and camera poses do not share the declared source clock')
            if not (np.issubdtype(dtype,np.integer) or dtype==bool):raise ValueError('Project identities or Boolean foreground, never intensity values')
            def frames():
                for t,page in enumerate(opened.pages):
                    image=page.asarray()
                    if image.dtype==bool:image=image.astype(np.uint8)
                    projected=to_native(image,matrices[t+offset],shape)
                    if not set(np.unique(projected)).issubset(set(np.unique(image))|{0}):raise AssertionError('Mask projection created a new identity')
                    yield projected.astype(dtype,copy=False)
            tifffile.imwrite(output,frames(),shape=(count,*shape),dtype=dtype,photometric='minisblack',
                bigtiff=True,compression='zlib',compressionargs={'level':1},metadata={'axes':'TYX'})
        records[name]=dict(path=str(output.resolve()),sha256=sha256(output),input=str(path.resolve()),
            input_sha256=sha256(path),source_frame_offset=offset,frames=count)
    report=dict(method_version=METHOD_VERSION,transforms=str(Path(transforms).resolve()),
        transforms_sha256=sha256(transforms),native_shape=list(shape),outputs=records,
        sampling='nearest identity at every original camera pixel centre; intensity images untouched')
    (target/'projection.json').write_text(json.dumps(report,indent=2))
    return report
