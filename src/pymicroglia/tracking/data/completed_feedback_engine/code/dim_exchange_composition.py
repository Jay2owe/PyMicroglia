"""Original departing/dim-name, smaller-flicker, satellite and exchange rules."""
import numpy as np
import restore_departing_soma_from_shared_outline as departing
import recover_displaced_soma_from_dim_name_reuse as reuse
import recover_smaller_bookended_owner_flickers as smaller
import recover_nearby_dim_temporal_satellites as satellite
import restore_spatial_reciprocal_soma_paths_a002 as exchange
import restore_moving_one_frame_soma_names as moving

def _sum(table, column):
    return int(table[column].sum()) if len(table) else 0

def _guard(labels,candidate,protected,audited,foreground):
    if (np.any((candidate != labels) & protected) or np.any((labels == 0) & (candidate > 0))
        or len(np.setdiff1d(np.unique(candidate),np.unique(labels))) or int((candidate != labels).sum()) != audited
        or (foreground and not np.array_equal(labels > 0,candidate > 0))):
        raise AssertionError('Original dim/exchange protection, foreground, names or pixel accounting failed')

def apply(labels,ledger,raw,points,restored):
    protected=labels != restored
    interim,audit1,apps1=departing.correct(labels,raw,points,protected)
    candidate,audit2,apps2=reuse.correct(interim,raw,points,protected | (interim != labels))
    _guard(labels,candidate,protected,_sum(apps1,'changed_pixels')+_sum(apps2,'pixels'),False)
    labels=candidate
    yield 'departing_dim_reuse',labels,ledger.copy(),dict(departing_soma_audit=audit1,departing_soma_applications=apps1,
        dim_name_reuse_audit=audit2,dim_name_reuse_applications=apps2)
    protected=labels != restored
    candidate,audit,applications=smaller.correct(labels,raw,points,protected)
    _guard(labels,candidate,protected,_sum(applications,'component_area'),False)
    labels=candidate
    yield 'smaller_flicker',labels,ledger.copy(),dict(smaller_bookended_flicker_audit=audit,smaller_bookended_flicker_applications=applications)
    for stage,rule,prefix in [('dim_satellite',satellite,'nearby_satellite'),('spatial_exchange',exchange,'spatial_exchange')]:
        protected=labels != restored
        candidate,audit,applications=rule.correct(labels,raw,points,protected)
        _guard(labels,candidate,protected,_sum(applications,'changed_pixels'),True)
        labels=candidate
        yield stage,labels,ledger.copy(),{prefix+'_audit':audit,prefix+'_applications':applications}
    protected=labels != restored
    candidate,audit,applications=moving.correct(labels,raw,protected)
    _guard(labels,candidate,protected,_sum(applications,'changed_pixels'),True)
    yield 'moving_flicker',candidate,ledger.copy(),dict(moving_name_flicker_audit=audit,moving_name_flicker_applications=applications)
