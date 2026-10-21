"""Draw prepared tracking frames; encoding and display preparation live outside drawing."""
import textwrap
import numpy as np
from . import panels


def render_frames(frames, *, field_view, window_px, options):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgba
    from auto_organotypic.render.luts import colormap, paint, LUT_ANCHORS
    panels._applied('pyflash')
    side = int(options['frame_px'])
    figure = Figure(figsize=(side/150,side/150),dpi=150)
    FigureCanvasAgg(figure)
    ax = figure.add_axes([.04,.13,.92,.75])
    anchored = options['cell_lut'] in LUT_ANCHORS
    pixels = np.zeros((window_px,window_px)) if anchored else np.zeros((window_px,window_px,3),np.uint8)
    picture = ax.imshow(pixels,cmap=colormap(options['cell_lut']) if anchored else None,
                        vmin=0,vmax=1,interpolation='nearest')
    blank = np.zeros((window_px,window_px,4))
    other = ax.imshow(blank.copy(),interpolation='nearest')
    own = ax.imshow(blank.copy(),interpolation='nearest')
    ax.set_axis_off()
    title = figure.text(.5,.935,'',ha='center',va='center',fontsize=11,color=panels.colour('circadian_ink'))
    caption = figure.text(.5,.055,'',ha='center',va='center',fontsize=9,color=panels.colour('orange'))
    marker = None
    if field_view is not None:
        inset = ax.inset_axes([.70,.70,.29,.29])
        inset.imshow(field_view,cmap='Greys',vmin=0,vmax=1.6,interpolation='nearest')
        marker = Rectangle((0,0),window_px,window_px,fill=False,
                           edgecolor=panels.colour('cyan'),linewidth=1)
        inset.add_patch(marker)
        inset.set_xticks([])
        inset.set_yticks([])
    try:
        for frame in frames:
            picture.set_data(frame['picture'] if anchored else paint(frame['picture'],options['cell_lut']))
            pixels = blank.copy()
            pixels[frame['other']] = to_rgba(panels.colour('muted'),.55)
            other.set_data(pixels)
            pixels = blank.copy()
            pixels[frame['own']] = to_rgba(panels.colour('cyan'))
            own.set_data(pixels)
            title.set_text(frame['heading'])
            caption.set_text(textwrap.fill(frame['caption'],46))
            caption.set_color(panels.colour('orange') if frame['missing'] else panels.colour('circadian_ink'))
            if marker is not None:
                centre = frame['centre']
                marker.set_xy((centre[1]-window_px//2,centre[0]-window_px//2))
            figure.canvas.draw()
            yield np.asarray(figure.canvas.buffer_rgba())[...,:3].copy()
    finally:
        figure.clear()


def follow(run, *, stem=None, identities=(), events=(), cell_count=6, span='recording',
           event_hours=6.0, fps=6, crop_px=None, locator=True, cell_lut=None,
           display_options=None, output_dir=None, dry_run=False, keep_frames=False,
           event_frames=(), frame_range=None):
    """Render saved identities over a recording, lifespan or event window."""
    from ..figure_tables.film_action import follow as action
    return action(run,stem=stem,identities=identities,events=events,cell_count=cell_count,span=span,
                  event_hours=event_hours,fps=fps,crop_px=crop_px,locator=locator,cell_lut=cell_lut,
                  display_options=display_options,output_dir=output_dir,dry_run=dry_run,keep_frames=keep_frames,
                  event_frames=event_frames,frame_range=frame_range)
