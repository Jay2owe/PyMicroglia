"""Prepare tracking films without changing identities or event decisions."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULTS: dict = {'span': 'recording', 'fps': 6, 'frame_px': 640, 'crop_px': 0, 'crop_margin_px': 8, 'locator': True, 'neighbours': True, 'event_hold_frames': 6, 'image_filter': 'auto-organotypic', 'display_black_percentile': 50.0, 'display_white_percentile': 99.8, 'display_gamma': 0.7, 'display_gain': 1.0, 'display_spatial_sigma': 1.0, 'display_pool_px': 4.0, 'display_sharpness': 3.0, 'display_noise_multiple': 1.0, 'display_pad_frames': 64, 'cell_lut': 'dluc_purple', 'outline': 'outline'}

@dataclass
class Film:
    """One cell's film and everything a reader needs to check it."""
    identity: int
    stem: str
    frames: list[int]
    hours: np.ndarray
    named: np.ndarray
    window_px: int
    centres: np.ndarray
    events: pd.DataFrame
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    display: dict = field(default_factory=dict)

def _plan(cell_frame: pd.DataFrame, identity: int, labels, options: dict) -> tuple:
    """Which frames the film covers, where the window sits, and how big it is."""
    cell = cell_frame[cell_frame['identity'] == identity].sort_values('frame_index')
    if cell.empty:
        raise ValueError(f"cell {identity} is not in this run's cell_frame table")
    n_frames = int(labels.shape[0])
    first, last = (int(cell['frame_index'].iloc[0]), int(cell['frame_index'].iloc[-1]))
    span = str(options['span']).strip().lower()
    if span == 'recording':
        frames = list(range(n_frames))
    elif span == 'lifespan':
        frames = list(range(first, last + 1))
    else:
        raise ValueError('--span must be recording or lifespan')
    half = int(options['crop_px']) // 2
    if not half:
        reach = 0
        for frame_index in range(first, last + 1):
            rows, columns = np.nonzero(labels[frame_index] == identity)
            if rows.size:
                reach = max(reach, int(np.ptp(rows)) // 2 + 1, int(np.ptp(columns)) // 2 + 1)
        half = reach + int(options['crop_margin_px'])
    window = 2 * half + 1
    known = {int(row.frame_index): (float(row.centroid_y), float(row.centroid_x)) for row in cell.itertuples()}
    if not known:
        raise ValueError(f'cell {identity} has no centroid to follow')
    centres, named = ([], [])
    held = known[first]
    for frame_index in frames:
        held = known.get(frame_index, held)
        centres.append(held)
        named.append(frame_index in known)
    return (frames, np.array(centres, dtype=float), np.array(named, dtype=bool), window)

def _region(centres: np.ndarray, window: int, shape) -> tuple[int, int, int, int]:
    """The rectangle every one of this cell's windows falls inside.

    The display filter is prepared over this, not over the whole field. Two
    reasons, and the second is the one that matters: a filter run over the
    field costs a hundred times as much for pixels no frame of the film shows,
    and the contrast range would then be set by tissue the cell never went
    near - which is the report card's reasoning as well.
    """
    half = window // 2 + 1
    top = max(int(np.floor(centres[:, 0].min())) - half, 0)
    left = max(int(np.floor(centres[:, 1].min())) - half, 0)
    bottom = min(int(np.ceil(centres[:, 0].max())) + half, shape[0])
    right = min(int(np.ceil(centres[:, 1].max())) + half, shape[1])
    return (top, bottom, left, right)

def _presentation(raw, labels, cell_frame: pd.DataFrame, options: dict, region: tuple[int, int, int, int]):
    """The display copy of one cell's own region, prepared once for its whole film."""
    from . import images as intensity
    if raw is None:
        return (None, {'image_filter': 'outlines only, no signal', 'display_only': True})
    offset = 0
    if {'source_imagej_frame', 'imagej_frame'} <= set(cell_frame.columns):
        offset = int(cell_frame['source_imagej_frame'].iloc[0] - cell_frame['imagej_frame'].iloc[0])
    top, bottom, left, right = region
    aligned = raw[offset:offset + labels.shape[0], top:bottom, left:right]
    return intensity.presentation_stack(aligned, image_filter=options['image_filter'], black_percentile=options['display_black_percentile'], white_percentile=options['display_white_percentile'], gamma=options['display_gamma'], time_gain=options['display_gain'], spatial_sigma_px=options['display_spatial_sigma'], pool_px=options['display_pool_px'], sharpness=options['display_sharpness'], noise_multiple=options['display_noise_multiple'], pad_frames=options['display_pad_frames'])

def _window(image: np.ndarray, centre, window: int, fill=0.0) -> np.ndarray:
    """One frame's window, padded rather than shifted where it leaves the field.

    Sliding the window back inside the field would be the one failure this
    whole file exists to avoid: the cell would stop being in the middle without
    anything on the frame saying so, and a cell at the edge is exactly the cell
    whose ending is being judged.
    """
    half = window // 2
    top, left = (int(round(centre[0])) - half, int(round(centre[1])) - half)
    out = np.full((window, window), fill, dtype=float)
    y0, x0 = (max(top, 0), max(left, 0))
    y1 = min(top + window, image.shape[0])
    x1 = min(left + window, image.shape[1])
    if y1 > y0 and x1 > x0:
        out[y0 - top:y1 - top, x0 - left:x1 - left] = image[y0:y1, x0:x1]
    return out

def _captions(events: pd.DataFrame | None, identity: int, hold: int) -> dict[int, str]:
    """What to write over which frames, in the figures' own wording."""
    if events is None or events.empty:
        return {}
    from .event_labels import event_label
    mine = events[events['identity'] == identity]
    written: dict[int, str] = {}
    for row in mine.itertuples():
        text = event_label(row.event)
        partner = getattr(row, 'partner_identity', None)
        if partner is not None and (not pd.isna(partner)):
            text += f' (cell {int(partner)})'
        for step in range(int(hold)):
            written.setdefault(int(row.frame_index) + step, text)
    return written
