"""Keep tracker-core evidence on its strongest-overlapping connected silhouette."""
import numpy as np
from scipy import ndimage as ndi

def component_owners(field):
    owners=np.zeros(field.shape,np.int32);offset=0
    for identity,roi in enumerate(ndi.find_objects(field),start=1):
        if roi is None:continue
        cc,count=ndi.label(field[roi]==identity,np.ones((3,3),bool))
        view=owners[roi];view[cc>0]=cc[cc>0]+offset;offset+=count
    return owners

def owned_core_support(support,owners, *, require_foreground_majority=False, centre=None, prefer_centred_owner=False):
    """Compare fractions of the same core: denominators cancel; retain ties."""
    centred = centre is not None and 0 <= centre[0] < owners.shape[0] and 0 <= centre[1] < owners.shape[1] and owners[centre] > 0
    if prefer_centred_owner and centred:
        return support & ((owners==0) | (owners==owners[centre]))
    if require_foreground_majority and not centred and np.count_nonzero(support & (owners>0))*2 <= np.count_nonzero(support):
        return np.zeros_like(support)
    names,counts=np.unique(owners[support & (owners>0)],return_counts=True)
    if len(names)<2:return support
    winners=names[counts==counts.max()]
    return support & ((owners==0)|np.isin(owners,winners))
