"""The memory comparison may defer display, never missing scientific output."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import tifffile

spec=importlib.util.spec_from_file_location('tracking_memory_comparison',
    Path(__file__).resolve().parents[1]/'compare_tracking_memory.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pair(tmp_path):
    roots=[tmp_path/'before',tmp_path/'after']
    for root in roots:
        final=root/'m23_completed_feedback/profile'
        final.mkdir(parents=True)
        (final/'run.json').write_text(json.dumps({'stages':[{'hash':'same','seconds':1}]}))
        tifffile.imwrite(final/'labels.tif',np.ones((3,4),np.uint16))
    return roots


def test_missing_science_is_never_deferred(tmp_path):
    a,b=pair(tmp_path)
    (b/'m23_completed_feedback/profile/labels.tif').unlink()
    assert module.compare(a,b,deferred_native_review=True)['status']=='fail'


def test_native_display_requires_explicit_deferral(tmp_path):
    a,b=pair(tmp_path)
    folder=a/'m6_review/profile';folder.mkdir(parents=True)
    tifffile.imwrite(folder/'review.tif',np.zeros((3,4),np.uint8))
    assert module.compare(a,b)['status']=='fail'
    report=module.compare(a,b,deferred_native_review=True)
    assert report['status']=='pass'
    assert report['deferred_native_review_files']==[str(Path('m6_review/profile/review.tif'))]
    (folder/'events.csv').write_text('important scientific evidence\n')
    assert module.compare(a,b,deferred_native_review=True)['status']=='fail'


def test_equal_pixels_different_compression_and_changed_pixel(tmp_path):
    a,b=pair(tmp_path)
    image=b/'m23_completed_feedback/profile/labels.tif'
    values=np.ones((3,4),np.uint16)
    tifffile.imwrite(image,values,compression='zlib')
    assert module.compare(a,b)['status']=='pass'
    values[1,2]=2
    tifffile.imwrite(image,values,compression='zlib')
    assert module.compare(a,b)['status']=='fail'


def test_boundary_hash_difference_fails(tmp_path):
    a,b=pair(tmp_path)
    (b/'m23_completed_feedback/profile/run.json').write_text(
        json.dumps({'stages':[{'hash':'changed','seconds':2}]}))
    assert module.compare(a,b)['status']=='fail'
