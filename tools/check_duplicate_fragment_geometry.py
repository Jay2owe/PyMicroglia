"""Exact full-movie and adversarial duplicate-fragment cleanup comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('engine','adapter','labels','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--issue',choices=['008','009'],default='009')
    a=p.parse_args();platform.platform=lambda *a,**k:'Windows-fragment-proof'
    sys.path[:0]=[str(a.engine.resolve()/'code'),str(a.adapter.resolve())]
    import numpy as np
    import tifffile
    from scipy import ndimage as ndi
    import accepted_history as history
    import duplicate_fragment_geometry as reuse
    module=history._issue_module(getattr(history,'I'+a.issue),'m9_duplicate_cleanup')
    original=module.clean_duplicates
    rng=np.random.default_rng(395)
    before=after=0.;events=0;edge_events=0;checks=0
    def check(movie,cfg,edge=False):
        nonlocal before,after,events,edge_events,checks
        tick=time.perf_counter();old,old_rows=original(movie,cfg,2);before+=time.perf_counter()-tick
        tick=time.perf_counter();new,new_rows=module.clean_duplicates(movie,cfg,2);after+=time.perf_counter()-tick
        assert np.array_equal(old,new)
        assert old_rows.to_csv(index=False)==new_rows.to_csv(index=False)
        for column in old_rows:
            if old_rows[column].dtype.kind=='f':assert old_rows[column].to_numpy().tobytes()==new_rows[column].to_numpy().tobytes(),column
        events+=len(old_rows);checks+=1
        if edge:edge_events+=len(old_rows)
    with reuse.install(module) as stats:
        assert stats['supported']
        for case in range(90):
            movie=np.zeros((3,41,43),dtype=np.int64)
            for i in range(45):
                y,x=rng.integers(0,38),rng.integers(0,40)
                movie[:,y:y+3,x:x+3]=rng.choice([-3,1,2,5,100000])
            cfg=dict(minimum_secondary_area_px=1+case%5,minimum_primary_distance_px=float(case%7),
                maximum_neighbour_distance_px=float(case%20),temporal_dilation_px=case%5-1,
                minimum_neighbour_area_without_temporal_px=case%12)
            check(movie,cfg,True)
        # Every pairwise nearest distance has the same float64 bits as EDT.
        for shape in ((1,9),(9,1),(23,37),(471,471)):
            for _ in range(12):
                aidx=rng.choice(np.prod(shape),min(30,np.prod(shape)),replace=False)
                bidx=rng.choice(np.prod(shape),min(45,np.prod(shape)),replace=False)
                left,right=reuse.Pixels(shape,aidx),reuse.Pixels(shape,bidx)
                expected=ndi.distance_transform_edt(~np.asarray(left))[np.asarray(right)].min()
                actual=reuse.Distances(left)[right].min()
                assert np.float64(expected).tobytes()==np.float64(actual).tobytes()
        movie=tifffile.imread(a.labels)
        cfg=dict(minimum_secondary_area_px=8,minimum_primary_distance_px=12.,maximum_neighbour_distance_px=15.,
            temporal_dilation_px=3,minimum_neighbour_area_without_temporal_px=48)
        full_before,full_after=before,after
        check(movie,cfg)
        full_before=before-full_before;full_after=after-full_after
    assert edge_events>0 and module.clean_duplicates is original
    report=dict(status='pass',checks=checks,frames=len(movie),events=events,edge_events=edge_events,
        before_seconds=before,after_seconds=after,full_movie_before_seconds=full_before,full_movie_after_seconds=full_after,
        counters=stats,scope='Exact cleanup operator and distances; complete hierarchy stage still required',
        settings=cfg,sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),Path(reuse.__file__),Path(module.__file__))},input_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','settings')}),flush=True)


if __name__=='__main__':main()
