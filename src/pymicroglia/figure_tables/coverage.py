"""Prepare cumulative occupancy classes and their exact field fractions."""
import numpy as np
import pandas as pd
import tifffile
from .prepared import PreparedViews


def prepare(source,options):
    path=source.input_path('labels')
    if path is None:raise ValueError('negative-space needs the labels input recorded in the run manifest')
    labels=tifffile.imread(path)
    arrays=[labels>0];names=['named'];labels_for=['Named cell']
    unclaimed=source.input_path('unclaimed')
    if unclaimed is not None:
        arrays.append(tifffile.imread(unclaimed)>0);names.append('unclaimed');labels_for.append('Unclaimed foreground')
    if any(a.shape!=labels.shape for a in arrays):raise ValueError('Occupancy sources must share one frame-by-row-by-column shape')
    cumulative=np.logical_or.accumulate(np.stack(arrays),axis=1)
    codes=np.zeros(labels.shape,dtype=np.uint16)
    for bit,array in enumerate(cumulative):codes|=array.astype(np.uint16)<<bit
    classes=[]
    for display_code,code in enumerate(np.unique(codes)):
        members=[i for i in range(len(names)) if code & (1<<i)]
        title='Never occupied' if not members else labels_for[members[0]]+' only' if len(members)==1 and len(names)>1 else ' and '.join(labels_for[i] for i in members)
        classes.append(dict(display_code=display_code,source_code=int(code),source_keys='|'.join(names[i] for i in members),occupancy_class=title,colour={0:'raw',1:'teal',2:'orange',3:'circadian_purple'}[int(code)]))
    hours=np.arange(len(labels),dtype=float)*source.interval/60
    rows=[]
    for frame in range(len(labels)):
        occupied=1.0-float(np.count_nonzero(codes[frame]==0)/codes[frame].size)
        for klass in classes:
            pixels=int(np.count_nonzero(codes[frame]==klass['source_code']))
            rows.append(dict(view='all_frames',frame_index=frame,hours=float(hours[frame]),**klass,pixels=pixels,share=float(pixels/codes[frame].size),total_occupied_share=occupied))
    table=pd.DataFrame(rows)
    mode=options['coverage_view']
    if mode not in {'final','stages'}:raise ValueError('coverage_view must be final or stages')
    count=max(2,int(options['stages'])) if mode=='stages' else 1
    frames=np.unique(np.linspace(0,len(labels)-1,count).round().astype(int)) if mode=='stages' else np.array([len(labels)-1])
    backgrounds=None
    if options['overlay']:
        raw_path=source.input_path('raw')
        if raw_path is not None:
            raw=tifffile.imread(raw_path)
            if raw.ndim!=3 or raw.shape[1:]!=labels.shape[1:] or len(raw)<len(labels):raise ValueError('Raw stack and label stack do not match')
            backgrounds=raw[-len(labels):][frames]
    display=np.zeros_like(codes)
    for row in classes:display[codes==row['source_code']]=row['display_code']
    order=[r for r in classes if r['source_code']!=0]+[r for r in classes if r['source_code']==0]
    shares=np.array([table.loc[table.source_code.eq(row['source_code']),'share'].to_numpy() for row in order])
    return PreparedViews({'coverage':dict(table=table.loc[table.frame_index.isin(frames)].copy(),frames=frames,images=display[frames],backgrounds=backgrounds,classes=classes,hours=hours[frames]),
        'composition':dict(table=table,hours=hours,shares=shares,classes=order,total=table.drop_duplicates('frame_index').total_occupied_share.to_numpy())},
        wording=dict(subtitle=f'Final field coverage: {np.count_nonzero(codes[-1])/codes[-1].size:.1%}.',
            footnote='Named cells have assigned identities. Unclaimed foreground has none. A mixed pixel was occupied by both sources at different times.')),None
