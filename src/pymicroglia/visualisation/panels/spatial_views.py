"""Selectable spatial maps and evidence views, receiving prepared encodings."""
import numpy as np
from . import spatial,colour
from ._contract import Drawn


def field(figure,data,*,style,**options):
    table,cfg=data['table'],data['cfg']
    snapshots=cfg['snapshots']
    columns=min(3,len(snapshots))
    axes=figure.subplots(int(np.ceil(len(snapshots)/columns)),columns,squeeze=False)
    for ax,snapshot in zip(axes.flat,snapshots):
        rows=table.loc[table.tile.eq(snapshot['tile'])]
        handle=spatial.field_map(ax,rows.loc[rows.record.eq('pixel')],shape=cfg['shape'],scale=cfg['scale'],
            cmap=cfg['cmap'],vmin=cfg['vmin'],vmax=cfg['vmax'])
        spatial._map_axes(ax,cfg)
        ax.set_title(f"{snapshot['hours']:g} h")
        arrows=rows.loc[rows.record.eq('arrow')]
        if not arrows.empty:
            ax.quiver(arrows.x,arrows.y,arrows.x2-arrows.x,arrows.y2-arrows.y,angles='xy',
                      scale_units='xy',scale=1,color=colour('black'))
        figure.colorbar(handle,ax=ax,label=cfg['value_label'],shrink=.7)
    for ax in list(axes.flat)[len(snapshots):]:
        ax.set_visible(False)
    return Drawn(table,axes)


def _period_maps(figure,data,key):
    table,cfg=data['table'],data['cfg']
    tiles=[tile for tile in cfg['tiles'] if (tile['key']=='period')==(key=='period')]
    axes=figure.subplots(1,len(tiles),squeeze=False)
    for ax,tile in zip(axes.flat,tiles):
        cells=table.loc[table.record.eq('cell')&table.tile.eq(tile['key'])]
        if key=='period' and cfg.get('spatial_tracks',True):
            spatial.track_map(ax,table.loc[table.record.eq('track')],cmap=tile['cmap'],
                              vmin=tile['vmin'],vmax=tile['vmax'],linewidth=cfg['spatial_track_width'])
        handle=spatial.cell_map(ax,cells,cmap=tile['cmap'],vmin=tile['vmin'],vmax=tile['vmax'],
                               point_size=cfg['point_size'],annotate=cfg['annotate'])
        spatial._map_axes(ax,cfg)
        ax.set_title(tile['title'])
        figure.colorbar(handle,ax=ax,label=tile['label'],shrink=.7)
    return Drawn(table,axes)


def period(figure,data,*,style,**options):
    return _period_maps(figure,data,'period')


def phase(figure,data,*,style,**options):
    return _period_maps(figure,data,'phase')


def positions(ax,data,*,style,**options):
    table,cfg=data['table'],data['cfg']
    spatial.cell_map(ax,table,value='spatial_order',cmap='viridis',vmin=1,vmax=max(2,len(table)),
                     point_size=cfg['point_size'],annotate=cfg['annotate'])
    spatial._map_axes(ax,cfg)
    ax.set_title(f"Order along {cfg['spatial_axis'].upper()} position")
    return Drawn(table,ax)


def matrix(ax,data,*,style,**options):
    table,cfg=data['table'],data['cfg']
    cells=table.loc[table.record.eq('cell')].sort_values(['spatial_order','identity'])
    handle=spatial.spatial_matrix(ax,table.loc[table.record.eq('trace')],order=cells.identity.tolist(),
        hours=cfg['hours'],cmap=cfg['cmap'],vmin=cfg['vmin'],vmax=cfg['vmax'],discrete=not cfg.get('show_history',False))
    ax.figure.colorbar(handle,ax=ax,label=cfg['value_label'],shrink=.7)
    ax.set(xlabel='Recorded hours',ylabel='Cell identity, in spatial order')
    return Drawn(table,ax)


def timing(ax,data,*,style,**options):
    table,cfg=data['table'],data['cfg']
    handle=spatial.cell_map(ax,table,cmap=cfg['cmap'],vmin=cfg['vmin'],vmax=cfg['vmax'],
                            point_size=cfg['point_size'],annotate=cfg['annotate'])
    spatial._map_axes(ax,cfg)
    ax.figure.colorbar(handle,ax=ax,label='Peak delay (h): positive means selected metric peaks later',shrink=.7)
    return Drawn(table,ax)


def connections(ax,data,*,style,**options):
    table,cfg=data['table'],data['cfg']
    pairs=table.loc[table.record.eq('pair')]
    edges=pairs.loc[pairs.neighbour.astype(str).str.lower().isin(['true','1','1.0'])]
    handle=spatial.connection_map(ax,edges,cmap=cfg['cmap'])
    nodes=table.loc[table.record.eq('cell')]
    ax.scatter(nodes.x,nodes.y,color=colour('dark'),s=45)
    spatial._map_axes(ax,cfg)
    ax.figure.colorbar(handle,ax=ax,label='Correlation: −1 opposite, +1 together',shrink=.7)
    return Drawn(table,ax)


def null(ax,data,*,style,**options):
    table,cfg=data['table'],data['cfg']
    ax.bar(table.x,table.value,width=table.x2-table.x,align='edge',color=colour('raw'))
    if cfg['contrast'] is not None:
        ax.axvline(cfg['contrast'],color=colour(cfg['line_color']),label='Measured difference')
    ax.set(xlabel='Mean neighbour correlation minus mean non-neighbour correlation',
           ylabel='Shuffled spatial arrangements',title=cfg['test_label'])
    ax.legend(frameon=False)
    return Drawn(table,ax)


def coverage(ax,data,*,style,**options):
    table,cfg=data['table'],data['cfg']
    ax.plot(table.hours,table.value*100,color=colour(cfg['line_color']),marker='o')
    ax.set(xlabel='Recorded hours',ylabel='Occupied share of ever-occupied tissue (%)',ylim=(0,100))
    return Drawn(table,ax)
