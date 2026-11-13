"""Validate newly allocated identities against exclusive prior physical support."""
import numpy as np
from scipy import ndimage as ndi

STRUCTURE=np.ones((3,3),bool)

def _supported_by_body(name_mask,body):
    """Every named pixel must be inside or connected to this body's same-frame mask."""
    for frame in range(len(body)):
        outside=name_mask[frame]&~body[frame]
        if not outside.any():continue
        connected=ndi.binary_propagation(body[frame],structure=STRUCTURE,mask=body[frame]|name_mask[frame])
        if np.any(outside&~connected):return False
    return True

def existing_owner_claims(before,allocated,points):
    if before.shape!=allocated.shape:raise ValueError('Aligned allocation movies required')
    if not np.array_equal(before>0,allocated>0):raise ValueError('Allocation must preserve foreground')
    observed=points[points.state.eq('observed')&points.strong.astype(bool)]
    f=observed.frame.to_numpy(int);y=np.floor(observed.y+.5).astype(int).to_numpy();x=np.floor(observed.x+.5).astype(int).to_numpy()
    if not ((f>=0)&(f<len(before))&(y>=0)&(y<before.shape[1])&(x>=0)&(x<before.shape[2])).all():raise ValueError('Measured points outside movie')
    prior_names=before[f,y,x];new_names=allocated[f,y,x];proposed={};audit=[]
    for new in np.setdiff1d(np.unique(allocated),np.unique(before)):
        body=allocated==new;choices=[]
        for old in np.unique(before[body]):
            if old==0:continue
            measured=(prior_names==old)&(new_names==new)
            if not measured.any():continue
            if _supported_by_body(before==old,body):choices.append(int(old))
        row=dict(allocated=int(new),supported_prior_names=choices,decision='unique_existing_owner'if len(choices)==1 else 'ambiguous_or_no_existing_owner')
        if len(choices)==1:proposed[int(new)]=choices[0]
        audit.append(row)
    conflicting={old for old in proposed.values()if list(proposed.values()).count(old)>1}
    claims={new:old for new,old in proposed.items()if old not in conflicting}
    for row in audit:
        if row['allocated']in proposed and proposed[row['allocated']]in conflicting:row['decision']='competing_new_bodies'
    return claims,audit

def compose_existing_owners(parent,allocation,claims):
    """Preserve current geometry; require every renamed branch to remain on the certified body."""
    if parent.shape!=allocation.shape:raise ValueError('Aligned parent and allocation required')
    result=parent.copy();audit=[]
    for new,old in claims.items():
        body=allocation==new;current=parent==new;target=parent==old
        supported=_supported_by_body(current,body)and _supported_by_body(target,body|current)
        row=dict(allocated=int(new),existing=int(old),pixels=int(current.sum()),decision='applied'if supported else 'independent_current_holder_or_branch')
        if supported:result[current]=old
        audit.append(row)
    assert np.array_equal(result>0,parent>0)
    return result,audit
