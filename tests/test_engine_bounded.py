"""The bounded runner calls every frozen numerical stage with the same inputs."""
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np
import tifffile
from pymicroglia.tracking.engine_bounded import run

def test_numerical_stages_and_read_only_pinned_arrays(tmp_path):
    path=tmp_path/'raw.tif';data=np.arange(72,dtype=np.uint16).reshape(2,6,6)
    tifffile.imwrite(path,data,compression='zlib',photometric='minisblack')
    config=SimpleNamespace(stems=['sample'])
    common=SimpleNamespace(Config=SimpleNamespace(load=lambda _:config))
    calls=[];threads={};base=np.zeros((4,6,6),np.uint16);base[:,2:4,2:4]=1
    flag=base>0
    pipeline=SimpleNamespace(ROOT=tmp_path/'engine',AcceptedHistoryInputs=lambda:object(),_pinned_paths=lambda *_:{'raw':path})
    def stage(name,result):
        def call(*args,**kwargs):
            calls.append(name)
            threads[name]=threading.get_ident()
            a=tifffile.imread(path);np.testing.assert_array_equal(a,data)
            assert not a.flags.writeable
            if name=='m22':
                np.testing.assert_array_equal(args[4],base)
                assert kwargs['m20_result_future'].done()
                np.testing.assert_array_equal(kwargs['upstream_inferred'],flag)
                folder=pipeline.ROOT/'m22_accepted_history'/'run'
                folder.mkdir(parents=True)
            return result
        return call
    functions={'stage_m0':('m0',{'raw':path}), 'stage_m1':('m1',None),
        'stage_m2':('m2',(base.copy(),'table',1,None,None)),
        'stage_m3':('m3',(base.copy(),None)),
        'stage_m4':('m4',(base.copy(),'events',flag.copy(),flag.copy())),
        'stage_m5':('m5',None),'stage_m7_accepted':('m7',(base.copy(),None)),
        'stage_m8_accepted':('m8',(base.copy(),flag.copy(),None)),
        'stage_m9_accepted':('m9',(base.copy(),flag.copy(),None)),
        'stage_m12_accepted':('m12',(base.copy(),flag.copy(),None)),
        'stage_m19_accepted':('m19',(base.copy(),flag.copy(),None)),
        'stage_m20_accepted':('m20',(base.copy(),flag.copy(),None)),
        'stage_m22_accepted_history':('m22',None)}
    for method,(name,result) in functions.items():setattr(pipeline,method,stage(name,result))
    original=tifffile.imread
    run(pipeline,common,'run','sample',tmp_path/'config.json')
    assert calls==['m0','m1','m2','m3','m4','m5','m7','m8','m9','m12','m19','m20','m22']
    assert threads['m20']!=threads['m19'] and threads['m22']==threads['m19']
    assert tifffile.imread is original
    assert not list(tmp_path.glob('motion-read-*'))
    assert (pipeline.ROOT/'m22_accepted_history/run/review_pending.json').is_file()
