"""Run the frozen numerical stages sequentially, without keeping obsolete movies.

Input pixels and stages are unchanged. Compressed pinned inputs are decoded once
to temporary file-backed arrays. The engine's whole-movie RGB review is deferred
to the streaming PyMicroglia renderer; this function returns tracking files only.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import gc, json, tempfile, weakref
from pathlib import Path
import tifffile

def run(pipeline, common, run_name, stem, config_path,*,array_pool=None,resume=False):
    cfg=common.Config.load(config_path)
    stages=("m0_inputs","m1_motion","m2_anchor","m3_track","m4_events","m5_reconcile","m7_persistence","m8_merge","m9_alias","m12_identity_reservation","m19_layered_lineage","m20_oscillatory_lineage","m22_accepted_history","m6_review")
    if stem not in cfg.stems:
        raise ValueError(f"stem {stem!r} is not configured")
    if not resume and any((pipeline.ROOT/stage/run_name).exists() for stage in stages):
        raise FileExistsError(f"immutable Motion run already exists: {run_name}")
    inputs={path.resolve() for path in pipeline._pinned_paths(cfg,stem).values()}
    original_read=tifffile.imread
    maps={}
    restored_maps=[]
    with tempfile.TemporaryDirectory(prefix="motion-read-") as scratch_name:
        scratch=Path(scratch_name)
        if array_pool is not None:
            (pipeline.ROOT/'decoded_storage.json').write_text(json.dumps(dict(temporary_directory=scratch_name),indent=2))
        def mapped_read(file,*args,**kwargs):
            try:key=Path(file).resolve()
            except TypeError:return original_read(file,*args,**kwargs)
            if key not in inputs or args or kwargs:
                return original_read(file,*args,**kwargs)
            if key not in maps:
                with tifffile.TiffFile(key) as opened:
                    maps[key]=opened.series[0].asarray(out=str(scratch/f"input_{len(maps)}.dat"))
                    if array_pool is not None:maps[key]=array_pool.adopt(maps[key],Path(maps[key].filename))
                    maps[key].flags.writeable=False
            return maps[key]
        tifffile.imread=mapped_read
        try:
            checkpoints=None
            if resume:
                from .checkpoints import Stages
                import uuid
                def restored_read(path):
                    with tifffile.TiffFile(path) as opened:
                        mapping=opened.series[0].asarray(out=str(scratch/(uuid.uuid4().hex+'.dat')))
                    restored_maps.append(weakref.ref(mapping))
                    return mapping if array_pool is None else array_pool.adopt(mapping,Path(mapping.filename))
                checkpoints=Stages(pipeline,cfg,stem,run_name,restored_read)
            def call(method,*args,**kwargs):
                return (checkpoints.call(method,*args,**kwargs) if checkpoints is not None
                        else getattr(pipeline,method)(*args,**kwargs))
            sources=call('stage_m0',cfg,stem,run_name)
            call('stage_m1',cfg,stem,run_name,sources)
            # This large five-channel display input can be decoded again if a
            # later stage requests it. Aliases retain their mapping lease.
            if array_pool is not None and 'motion_composite' in sources:
                maps.pop(Path(sources['motion_composite']).resolve(),None)
            observations,table,anchor,_,_=call('stage_m2',cfg,stem,run_name,sources)
            initial,_=call('stage_m3',cfg,stem,run_name,sources,observations,table,anchor)
            labels,events,inferred,unresolved=call('stage_m4',cfg,stem,run_name,sources,initial)
            del initial;gc.collect()
            call('stage_m5',cfg,stem,run_name,sources,labels,events,inferred)
            persistent,_=call('stage_m7_accepted',cfg,stem,run_name,sources,labels,observations,table,anchor)
            del labels;gc.collect()
            merged,merge_inferred,_=call('stage_m8_accepted',cfg,stem,run_name,sources,persistent)
            del persistent;gc.collect()
            aliased,alias_inferred,_=call('stage_m9_accepted',cfg,stem,run_name,sources,merged)
            del merged;gc.collect()
            history_inputs=pipeline.AcceptedHistoryInputs()
            reserved,reservation_inferred,_=call('stage_m12_accepted',cfg,stem,run_name,sources,aliased,observations,table,anchor,history_inputs=history_inputs)
            del aliased;gc.collect()
            final,lineage_inferred,_=call('stage_m19_accepted',cfg,stem,run_name,sources,reserved,observations,table,anchor,history_inputs=history_inputs)
            del reserved;gc.collect()
            pre_inferred=inferred|merge_inferred|alias_inferred|reservation_inferred|lineage_inferred
            del inferred,merge_inferred,alias_inferred,reservation_inferred,lineage_inferred;gc.collect()
            # Match the frozen runner's worker-thread numeric context. NumPy
            # 2.1 initialises scalar promotion differently in a fresh thread.
            # Wait here so the bounded runner still runs one heavy stage at a time.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future=executor.submit(call,'stage_m20_accepted',cfg,stem,run_name,sources,final.copy())
                future.result()
            call('stage_m22_accepted_history',cfg,stem,run_name,sources,final.copy(),observations,table,anchor,history_inputs=history_inputs,m20_result_future=future,upstream_inferred=pre_inferred,unresolved=unresolved)
            # New data have not been approved by a human; don't write inherited approval text.
            (pipeline.ROOT/"m22_accepted_history"/run_name/"review_pending.json").write_text(json.dumps(dict(status="pending",reason="Full-duration new-input tracking requires a source-linked review; full-field RGB preparation deferred to PyMicroglia."),indent=2))
            print(f"DONE {stem}: accepted engine stages completed; new-input review pending",flush=True)
        finally:
            tifffile.imread=original_read
            # Any aliases held by engine helpers are read-only and no longer used.
            for array in maps.values():
                mmap=getattr(array,"_mmap",None)
                if mmap is not None:mmap.close()
            for reference in restored_maps:
                array=reference()
                mmap=getattr(array,'_mmap',None)
                if mmap is not None and not mmap.closed:mmap.close()
            maps.clear();gc.collect()
