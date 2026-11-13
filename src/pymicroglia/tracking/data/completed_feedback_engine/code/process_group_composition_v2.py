"""Regenerate process-group measurements and path seeds from fresh ancestors."""
from io import StringIO
import numpy as np
import pandas as pd
import process_inventory_tracked as tracked
import process_inventory_temporal as temporal
import process_inventory_owners as owners
import process_inventory_intensity as intensity
import process_inventory_shape as shape
import process_inventory_interface_v2 as interface
import process_inventory_bridge as bridge
import process_inventory_motion as motion
import reunite_large_process_groups as groups
import extend_approved_process_paths as extension

INVENTORIES = [('tracked', tracked), ('temporal', temporal), ('owners', owners),
               ('intensity', intensity), ('shape', shape), ('interface', interface),
               ('bridge', bridge), ('motion', motion)]

def csv_boundary(table):
    try:
        return pd.read_csv(StringIO(table.to_csv(index=False)))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def inventory(movies):
    """Movie keys route arrays only; no key selects a correction or eligibility."""
    table = None
    for stage, module in INVENTORIES:
        if table is None or len(table):
            table = module.scan(movies, table)
        if not len(table) and 'well' not in table:
            # No eligible pairs is valid for an independent new recording.
            # Later inventories only refine these pairs; none can create one.
            table = table.assign(well=pd.Series(dtype=str))
        yield stage, table
        table = csv_boundary(table)

def seeds_from_applications(before, after, applications):
    result = []
    for row in csv_boundary(applications).itertuples(index=False):
        t, old, new = int(row.frame)-1, int(row.old_name), int(row.new_name)
        mask = (before[t] == old) & (after[t] == new)
        if not np.any(mask):
            raise AssertionError('Fresh process seed missing')
        result.append(dict(frame=t, large_name=new, old_name=old, mask=mask))
    return result

def apply(movie, neighbours):
    labels, raw, points = movie['branch_labels'], movie['raw'], movie['points']
    original, ledger = movie['restored'], movie['ledger']
    protected = labels != original
    if len(neighbours):
        candidate, evidence, audit, applications = groups.correct(labels, raw, points, protected, neighbours)
    else:
        candidate, evidence, audit, applications = labels.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    _guard(labels, candidate, protected, applications)
    yield 'process_groups', candidate, ledger.copy(), dict(process_group_frame_evidence=evidence,
        process_group_audit=audit, process_group_applications=applications)
    seeds = seeds_from_applications(movie['pre_branch_labels'], labels, movie['branch_applications'])
    seeds += seeds_from_applications(labels, candidate, applications)
    labels = candidate
    protected = labels != original
    if seeds:
        candidate, evidence, audit, applications = extension.correct(labels, raw, protected, seeds)
    else:
        candidate, evidence, audit, applications = labels.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    _guard(labels, candidate, protected, applications)
    yield 'process_extensions', candidate, ledger.copy(), dict(process_extension_frame_evidence=evidence,
        process_extension_audit=audit, process_extension_applications=applications)

def _guard(labels, candidate, protected, applications):
    count = int((labels != candidate).sum())
    applied = int(applications.changed_pixels.sum()) if len(applications) else 0
    if count != applied or not np.array_equal(candidate > 0, labels > 0) or np.any((candidate != labels) & protected):
        raise AssertionError('Original foreground/protection/application boundary violated')
