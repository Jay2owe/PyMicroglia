"""Real channel images beneath a spatial map, prepared once before drawing.

Each requested channel is reduced to one image in label coordinates (a 2-D
image is used as given; a stack is averaged over the frames matching the label
movie, the recording's last frames, as the coverage overlay does), scaled to
its own 1st-99.5th percentile and painted through its house channel map from
``auto_organotypic.render.luts``. Several channels are added, each at its
weight, and clipped: brightfield at half weight keeps fluorescence readable.
The map is then drawn over this picture at ``underlay_alpha``.
"""
import numpy as np
import pandas as pd
import tifffile

#: House map for a channel named in the request; other names need ``underlay_luts``.
CHANNEL_LUTS = {'brightfield': 'grays', 'phase': 'grays', 'red': 'red', 'green': 'green',
                'raw': 'dluc_purple', 'bioluminescence': 'dluc_purple'}
PERCENTILES = (1., 99.5)


def _requests(options):
    requested = options.get('underlay') or []
    if isinstance(requested, str):
        requested = [part for part in requested.split(',') if part.strip()]
    rows = []
    for item in requested:
        name, _, path = str(item).partition('=')
        name, path = name.strip(), path.strip()
        if not name:
            raise ValueError('Each underlay channel needs a name, for example brightfield=<image.tif> or raw')
        rows.append((name, path or None))
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError('Each underlay channel may be named once')
    return rows


def _per_channel(values, names, option):
    values = list(values or [])
    if not values:
        return None
    if len(values) == 1:
        return values * len(names)
    if len(values) != len(names):
        raise ValueError(f'{option} needs one value or one per underlay channel ({len(names)})')
    return values


def _image(source, name, path, shape, frames):
    resolved = path or source.input_path(name)
    if resolved is None:
        raise ValueError(f'Underlay channel {name!r} is not a recorded input; give it as {name}=<image.tif>')
    if path is not None and hasattr(source, 'sources'):
        source.sources.append(resolved)
    stack = np.asarray(tifffile.imread(resolved), dtype=float)
    used = 1
    if stack.ndim == 3:
        if frames is not None:
            if len(stack) < frames:
                raise ValueError(f'Underlay channel {name!r} has {len(stack)} frames; the labels have {frames}')
            stack = stack[-frames:]
        used = len(stack)
        stack = stack.mean(axis=0)
    if stack.shape != tuple(shape):
        raise ValueError(f'Underlay channel {name!r} is {stack.shape[1]} x {stack.shape[0]} pixels; '
                         f'the cell labels are {shape[1]} x {shape[0]}. Give the image in label coordinates')
    return stack, str(resolved), used


def screen(image, percentiles=PERCENTILES):
    """Scale one image to 0..1 between its own percentiles."""
    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, percentiles) if finite.size else (0., 1.)
    if high <= low:
        return np.zeros_like(image), float(low), float(high)
    return np.clip((image - low) / (high - low), 0, 1), float(low), float(high)


def underlay_rgb(images, luts, weights=None):
    """Paint each 2-D image through its map; several are added at their weights and clipped."""
    from auto_organotypic.render.luts import paint
    weights = weights or {}
    total = sum(weights.get(name, 1.) * paint(screen(image)[0], luts[name]).astype(float)
                for name, image in images.items())
    return np.clip(total, 0, 255).astype(np.uint8)


def prepare_underlay(source, options, shape, frames=None):
    """The painted underlay and one provenance row per channel, or ``None`` when none was requested."""
    requests = _requests(options)
    if not requests:
        return None
    names = [name for name, _ in requests]
    luts = _per_channel(options.get('underlay_luts'), names, 'underlay_luts')
    if luts is None:
        missing = [name for name in names if name.lower() not in CHANNEL_LUTS]
        if missing:
            raise ValueError(f'No house map for underlay channel(s) {", ".join(missing)}; set underlay_luts')
        luts = [CHANNEL_LUTS[name.lower()] for name in names]
    weights = [float(w) for w in (_per_channel(options.get('underlay_weights'), names, 'underlay_weights') or [1.] * len(names))]
    if any(not np.isfinite(w) or w < 0 for w in weights):
        raise ValueError('underlay_weights must be finite and not negative')
    alpha = float(options.get('underlay_alpha', .6))
    if not 0 < alpha <= 1:
        raise ValueError('underlay_alpha must be above 0 and at most 1')
    images, rows = {}, []
    for (name, path), lut, weight in zip(requests, luts, weights):
        image, resolved, used = _image(source, name, path, shape, frames)
        _, low, high = screen(image)
        images[name] = image
        rows.append(dict(channel=name, source=resolved, frames_averaged=used, lut=str(lut), weight=weight,
                         display_minimum=low, display_maximum=high, lower_percentile=PERCENTILES[0],
                         upper_percentile=PERCENTILES[1], map_alpha=alpha))
    rgb = underlay_rgb(images, dict(zip(names, luts)), dict(zip(names, weights)))
    return dict(rgb=rgb, alpha=alpha, table=pd.DataFrame(rows))
