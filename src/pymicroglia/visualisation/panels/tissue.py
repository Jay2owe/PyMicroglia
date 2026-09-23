"""Draw a prepared full-field map and its exact display range."""
from . import colour
from ._contract import Drawn
from matplotlib import colormaps


def spatial(ax,data,*,style=None,**options):
    if data.get('message'):
        ax.text(.5,.5,data['message'],ha='center',transform=ax.transAxes);ax.set_axis_off()
        return Drawn(data['table'],ax)
    field=data['field'];cmap=colormaps[data['cmap']].with_extremes(bad=colour('raw'))
    image=ax.imshow(data['array'],cmap=cmap,vmin=data['limits'][0],vmax=data['limits'][1],extent=(0,field['width'],field['height'],0),interpolation='nearest')
    for segment in data.get('outlines',[]):ax.plot(segment[:,0],segment[:,1],color=colour('ink'))
    if data.get('message'):ax.text(.5,.5,data['message'],ha='center',transform=ax.transAxes)
    ax.set(xlabel=f"X position ({data['unit']})",ylabel=f"Y position ({data['unit']})",aspect='equal')
    ax.figure.colorbar(image,ax=ax,label=data['label'])
    return Drawn(data['table'],ax)
