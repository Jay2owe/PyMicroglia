"""Prepare programme similarity from the saved complete pairwise distances."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster,leaves_list,dendrogram
from scipy.spatial.distance import squareform
from .regimes import _regime_rows,_regime_labels
from .prepared import PreparedViews

def prepare(source,options):
    regimes = _regime_rows(source.table('cell_frame.csv'))
    regime_names = _regime_labels(source.table('regime_profiles.csv'))
    pairs = source.table('sequence_distance.csv')
    window_hours = float(source.module_params('sequence_distance')['window_hours'])
    identities = sorted(set(pairs['identity_a'].astype(int)) | set(pairs['identity_b'].astype(int)))
    index = {identity: position for position, identity in enumerate(identities)}
    expected = len(identities) * (len(identities) - 1) // 2
    if len(pairs) != expected:
        raise ValueError(f'sequence_distance.csv holds {len(pairs)} pairs for {len(identities)} cells, not the {expected} every pair would make. The run filtered pairs out - check min_overlap_steps in the sequence_distance settings - and a pair this page cannot see would be drawn as two cells at no distance at all.')
    distances = np.zeros((len(identities), len(identities)), dtype=float)
    for row in pairs.itertuples():
        left_index, right_index = (index[int(row.identity_a)], index[int(row.identity_b)])
        distances[left_index, right_index] = distances[right_index, left_index] = row.dtw_distance
    data = pairs[['identity_a', 'identity_b', 'dtw_distance', 'alignment_offset_hours', 'alignment_overlap_steps']].copy().rename(columns={'dtw_distance': 'different_state_fraction', 'alignment_offset_hours': 'mean_alignment_offset_hours', 'alignment_overlap_steps': 'aligned_steps_with_both_cells_observed'})
    if len(identities) > 1:
        tree = linkage(squareform(distances, checks=False), method='average')
        cluster_count = max(1, min(int(options.get('clusters')), len(identities)))
        groups = fcluster(tree, cluster_count, criterion='maxclust')
        leaf_order = leaves_list(tree)
    else:
        tree = np.empty((0, 4))
        groups = np.ones(len(identities), dtype=int)
        leaf_order = np.arange(len(identities))
    cluster_map = dict(zip(identities, groups.astype(int)))
    if not data.empty:
        data['similarity_group_a'] = data['identity_a'].map(cluster_map)
        data['similarity_group_b'] = data['identity_b'].map(cluster_map)
    display_order=sorted(range(len(identities)),key=lambda index:(groups[index],list(leaf_order).index(index)))
    ordered=[identities[index] for index in display_order]
    pivot=regimes.pivot(index='identity',columns='hours',values='regime').reindex(ordered)
    ribbon_table=pivot.reset_index().melt('identity',var_name='hours',value_name='regime')
    branches=dendrogram(tree,no_plot=True,orientation='right') if len(tree) else {'icoord':[],'dcoord':[]}
    tree_table=pd.DataFrame([dict(branch=i,point=j,lane=y,distance=x) for i,(ys,xs) in enumerate(zip(branches['icoord'],branches['dcoord'])) for j,(y,x) in enumerate(zip(ys,xs))])
    return PreparedViews({'similarity':dict(table=data,matrix=distances[np.ix_(display_order,display_order)],rows=ordered,columns=ordered,label='Different aligned states (fraction)',vmin=0,vmax=1),
        'tree':dict(table=tree_table),'groups':dict(table=ribbon_table,matrix=pivot.to_numpy(float),margins=np.ones(pivot.shape),hours=pivot.columns.to_numpy(float),names=regime_names)},
        auxiliary={'sequence_pairs.csv':data},wording=dict(subtitle=f'{len(identities)} cells; original sequence alignment allowed {window_hours:g} hours of displacement.',
        footnote='Groups describe similarity of the saved state sequences. These are descriptive groups, not evidence of a common rhythm period.')),None
