"""Generate the accepted corrections' evidence from the current native run.

Must execute inside the native engine's import context. All paths are derived
from that run; the historical evaluation dataset is never read here.
"""
import importlib
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tifffile

from .completed_feedback import extract, methods, recipe


def _csv_boundary(table):
    if not len(table.columns):
        return table.copy()
    return pd.read_csv(io.StringIO(table.to_csv(index=False)), low_memory=False)


def _table(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def generate(root, stem, run_name, config_path, folder):
    """Return inputs for one movie, including newly computed tracker variants."""
    import accepted_history as ah
    import pipeline
    from common import Config
    from native_tracking_prefix import collect_tracking_prefix

    started = time.perf_counter()
    root = Path(root)
    cfg = Config.load(Path(config_path))
    offset = cfg.source_frame_offset
    sources = pipeline._pinned_paths(cfg, stem)
    native = root/'m22_accepted_history'/run_name
    history = native/'mid/accepted_history'
    lineage = root/'m19_layered_lineage'/run_name
    read = tifffile.imread
    labels = read(native/'out'/f'{stem}.tif')
    points = _table(history/'26_latent_body_tracks/out/latent_track_points.csv')
    raw = read(sources['registered_raw'])
    lag = read(sources['lag_float'])
    physical = read(lineage/'mid'/f'{stem}_physical_somas.tif')
    reservation = read(root/'m12_identity_reservation'/run_name/'out'/f'{stem}.tif')
    observations, observation_table, anchor = pipeline._load_anchor_inputs(stem, run_name)
    reserved = ah._field_motion_reservations(run_name)
    evidence = _table(lineage/'mid/comparison_evidence.csv')
    if 't' in evidence:
        evidence['t'] -= offset
    before_stationary = read(history/'12_issue023_generalized.tif')
    after_short = read(history/'13_issue027_short_blank_recovery.tif')
    # The stationary stage only renames existing foreground; short recovery
    # only adds foreground. This is the original accepted reconstruction.
    observed = np.where(before_stationary > 0, after_short, 0).astype(after_short.dtype, copy=False)
    del before_stationary, after_short
    data = dict(labels=labels,
        ledger=read(native/'out'/f'{stem}_unclaimed_original_ids.tif'),
        raw=raw[offset:offset+len(labels)], points=points,
        birth_labels=read(lineage/'mid'/f'{stem}_attempt005_comparison.tif'),
        births=_table(lineage/'out/comparison_links.csv'),
        presence_input=observed,
        presence_saved=read(history/'15_issue030_moderate_gap_recovery.tif'),
        presence_parameters=json.loads((history/'23_small_cell_blank_recovery/out/producer_metrics.json').read_text())['parameters'],
        recovery_history=pd.concat([_table(history/stage/'out/producer_audit.csv')
            for stage in ('22_short_blank_recovery','23_small_cell_blank_recovery','24_moderate_gap_recovery')], ignore_index=True),
        physical=physical[offset:offset+len(labels)], physical_evidence=evidence)
    final = {}
    timings = []
    corrections = extract(folder)
    settings = recipe(corrections)
    native_points = points.copy()
    native_points['frame'] += offset
    cache = {}
    tick = time.perf_counter()
    before, _, stages = collect_tracking_prefix(cfg,physical,reservation,raw,lag,
        observations,observation_table,anchor,reserved,points=native_points,prepared_cache=cache)
    timings.append(dict(stage='gap_reference',seconds=time.perf_counter()-tick,operations=stages))
    with methods(corrections,preserve_native=True):
        from recovery_sequence_composition import frozen_method
        for spec in settings['operations']['final']:
            tick = time.perf_counter()
            values = dict(points=points)
            with frozen_method(spec):
                if spec['kind'] == 'gap':
                    memory = importlib.import_module('measured_soma_gap_memory')
                    after, events, stages = collect_tracking_prefix(cfg,physical,reservation,raw,lag,
                        observations,observation_table,anchor,reserved,points=native_points,
                        rule_context=memory.measured_soma_gap_memory,prepared_cache=cache)
                    values['traces'] = [(old[0],new[0],_csv_boundary(new[1]))
                        for old,new in zip(before,after)]
                    before = after
                elif spec['kind'] == 'restoration':
                    memory = importlib.import_module('measured_restoration_continuity_a002')
                    old = read(history/'07_issue011.tif')
                    old_ledger = read(history/'07_issue011_unclaimed.tif')
                    protected = read(history/'06_issue010.tif')
                    aligned_lag = ah.align_transition_stack(lag,len(old),offset)
                    ownership = ah._issue_module(ah.I012,'m1_reconcile_domain_ownership')
                    detector = ah._issue_module(ah.I012,'m8_detect_dormant_releases')
                    reconciler = ah._issue_module(ah.I012,'m9_reconcile_dormant_releases')
                    params = ah._params('issue012','domain_ownership')
                    measurements = []
                    with memory.measured_restoration_continuity(ownership,points,measurements):
                        candidate,ledger,actions,_ = ownership.reconcile(old,old_ledger,aligned_lag,
                            set(params['methods']),params['ownership'],offset,old!=protected)
                    detect_params = ah._params('issue012','dormant_detection')
                    reconcile_params = ah._params('issue012','dormant_reconcile')
                    events,candidates = detector.detect(candidate,data['raw'],aligned_lag,actions,
                        offset,detect_params['detection'])
                    selected = reconciler.select_events(events,candidates,reconcile_params['reconcile'])
                    root_after,_,_ = reconciler.reconcile(candidate,ledger,data['raw'],selected,
                        offset,reconcile_params['reconcile'])
                    values.update(root_before=read(history/'08_issue012.tif'),root_after=root_after,
                        events=json.loads(json.dumps(measurements)))
                    # These ancestors/proposals are not inputs to later gap
                    # passes. Keep only the two restoration arrays in values.
                    del old, old_ledger, protected, candidate, ledger, aligned_lag
            final[spec['stage']] = values
            timings.append(dict(stage=spec['stage'],seconds=time.perf_counter()-tick))
    data['final_evidence'] = final
    timing_path = Path(folder)/f'{stem}_correction_evidence.json'
    timing_path.write_text(json.dumps(dict(source_run=str(native),source_frame_offset=offset,
        seconds=time.perf_counter()-started,stages=timings),indent=2)+'\n')
    return data
