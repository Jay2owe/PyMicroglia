"""Lazy native-pixel display and exact saved mask geometry."""
from __future__ import annotations
import base64
import hashlib
from functools import lru_cache
from contextlib import contextmanager, ExitStack
from io import BytesIO
import os
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from .exploration_data import manifest, recording
from pymicroglia.measure.mask_geometry import observation

MAX_PLANE_BYTES = 32 * 1024**2
MAX_HISTORY = 100
MAX_EXACT_HISTORY = 1000
MAX_CELL_STRIP_BYTES = 24 * 1024**2


@lru_cache(maxsize=1)
def image_luts():
    """The shared image/movie lookup tables, sampled at native display levels."""
    from auto_organotypic.render import luts
    levels = np.arange(256, dtype=np.float32) / 255
    return {"default": "grays", "source": "auto_organotypic.render.luts",
            "maps": [{"key": name,
                      "label": "Grayscale" if name == "grays" else name.replace("_", " ").capitalize(),
                      "table": luts.paint(levels, name).tolist()}
                     for name in ["grays", *[key for key in luts.lut_names() if key != "grays"]]]}


def declared_source(root, movie, role):
    value = movie.get("spec", {}).get(role) or movie.get(role)
    if not value:
        return None
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if resolved.is_file() or not path.is_absolute():
        return resolved
    # Saved runs can move together with their sibling raw/mask folders. Rebase
    # against the recorded run location, preserving that exact relationship.
    _, data = manifest(root)
    original_run = data.get("run", {}).get("folder")
    if not original_run or not Path(original_run).is_absolute():
        return resolved
    try:
        relocated = (root / os.path.relpath(path, original_run)).resolve()
    except ValueError:
        return resolved
    if not relocated.is_file():
        return resolved
    expected = movie.get("spec", {}).get("sha256", {}).get(role)
    if expected and _source_hash(*signature(relocated)) != expected:
        raise ValueError(f"The relocated {role} source does not match the saved recording.")
    return relocated


@lru_cache(maxsize=24)
def _source_hash(path, size, modified):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(path):
    if path is None or not path.is_file():
        raise ValueError("The declared source image is unavailable locally.")
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=12)
def _plane(path, size, modified, frame):
    with tifffile.TiffFile(path) as tif:
        return _read_plane(tif, path, frame)


def _read_plane(tif, path, frame):
    series = tif.series[0]
    shape, axes = series.shape, series.axes
    if len(shape) not in (2, 3) or axes[-2:] != "YX":
        raise ValueError(f"Display requires an aligned single-channel YX or TYX stack; saved axes are {axes}.")
    count = 1 if len(shape) == 2 else shape[0]
    if not 0 <= frame < count:
        raise ValueError("Requested frame is outside the declared source stack.")
    if np.prod(shape[-2:]) * series.dtype.itemsize > MAX_PLANE_BYTES:
        raise ValueError("This image plane exceeds the 32 MiB display budget.")
    if len(series.pages) == count:
        pixels = series.pages[frame].asarray()
    else:
        try:
            mapped = tifffile.memmap(path)
            pixels = np.array(mapped if len(shape) == 2 else mapped[frame])
            del mapped
        except ValueError:
            if np.prod(shape) * series.dtype.itemsize > MAX_PLANE_BYTES:
                raise ValueError("This compressed stack cannot supply a lazy plane within the display budget.") from None
            pixels = series.asarray()
            if len(shape) == 3:
                pixels = pixels[frame]
    if pixels.ndim != 2:
        raise ValueError("Source page does not have two spatial dimensions.")
    pixels.setflags(write=False)
    return pixels


def plane(path, frame):
    return _plane(*signature(path), int(frame))


@contextmanager
def _plane_reader(path):
    """One request owns the file handle and at most two decoded planes."""
    with ExitStack() as handles:
        tif = None
        @lru_cache(maxsize=2)
        def read(frame):
            nonlocal tif
            if tif is None:
                signature(path)  # Do not open unused raw data for absent cells.
                tif = handles.enter_context(tifffile.TiffFile(path))
            return _read_plane(tif, str(path), frame)
        try:
            yield read
        finally:
            read.cache_clear()


@lru_cache(maxsize=8)
def _contrast(path, size, modified):
    pixels = _plane(path, size, modified, 0)
    sample = pixels.ravel()[::max(1, pixels.size // 100_000)]
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return 0.0, 1.0
    lo, hi = np.percentile(sample, [1, 99.8])
    return float(lo), float(max(lo + 1, hi))


def png(pixels, contrast, *, preview=False):
    lo, hi = contrast
    if preview:
        # Bound temporary floating-point display work as well as encoded bytes.
        # Native detail uses every source pixel through the unchanged path below.
        stride = max(1, int(np.ceil(max(pixels.shape) / 1024)))
        pixels = pixels[::stride, ::stride]
    scaled = np.nan_to_num(np.clip((pixels.astype(float) - lo) * 255 / (hi - lo), 0, 255)).astype(np.uint8)
    out = BytesIO()
    picture = Image.fromarray(scaled)
    if preview:
        picture.thumbnail((256, 256), Image.Resampling.LANCZOS, reducing_gap=3)
        picture.save(out, format="JPEG", quality=65)
    else:
        picture.save(out, format="PNG")
    return ("data:image/jpeg;base64," if preview else "data:image/png;base64,") + base64.b64encode(out.getvalue()).decode("ascii")


def _change_image(current, previous, identity, crop):
    """Transparent display overlay of saved identity pixels gained and lost since frame - 1."""
    x0, y0, x1, y1 = crop
    now = current[y0:y1, x0:x1] == identity
    before = previous[y0:y1, x0:x1] == identity if previous is not None else np.zeros_like(now)
    overlay = np.zeros((*now.shape, 4), dtype=np.uint8)
    if previous is not None:
        overlay[now & ~before] = (114, 222, 211, 240)
    overlay[before & ~now] = (239, 199, 123, 240)
    out = BytesIO()
    Image.fromarray(overlay, "RGBA").save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def _padded_bounds(bounds, width, height, padding=15):
    x0, y0, x1, y1 = bounds
    return (max(0, x0 - padding), max(0, y0 - padding), min(width, x1 + padding), min(height, y1 + padding))


def _union_bounds(first, second):
    if first is None:
        return second
    if second is None:
        return first
    return (min(first[0], second[0]), min(first[1], second[1]),
            max(first[2], second[2]), max(first[3], second[3]))


def load_change(run, stem, identity, frame):
    """Exact selected-frame change for recordings too long to preload as cell crops."""
    root, _, movie = recording(run, stem)
    labels = declared_source(root, movie, "labels")
    current = plane(labels, frame)
    previous = plane(labels, frame - 1) if frame else None
    yy, xx = np.nonzero((current == identity) | ((previous == identity) if previous is not None else False))
    if not yy.size:
        return {"frame_index": frame, "identity": identity, "change": None, "change_bounds": None}
    crop = _padded_bounds((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1),
                          current.shape[1], current.shape[0])
    return {"frame_index": frame, "identity": identity, "change": _change_image(current, previous, identity, crop),
            "change_bounds": list(crop)}


def geometry(labels, identity=None):
    if labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Saved masks must contain nonnegative integer identities.")
    identities = [identity] if identity is not None else [int(i) for i in np.unique(labels) if i]
    if len(identities) > 1000:
        raise ValueError("This plane exceeds the 1,000-cell display budget.")
    # One pass finds tight boxes for ordinary label images. Do not allocate a
    # label-indexed table for arbitrary large IDs, or rescan the full field for
    # every cell. Coordinates inside each box retain the original row order.
    boxes = None
    if identity is None and identities and max(identities) <= 65536:
        from scipy.ndimage import find_objects
        boxes = find_objects(labels, max_label=max(identities))
    result = []
    for i in identities:
        box = boxes[i - 1] if boxes is not None else None
        if box is None:
            yy, xx = np.nonzero(labels == i)
        else:
            yy, xx = np.nonzero(labels[box] == i)
            yy, xx = yy + box[0].start, xx + box[1].start
        if yy.size:
            result.append({"identity": i, **observation(yy, xx)})
    return result


def load_frame(run, stem, frame):
    result = load_frame_image(run, stem, frame)
    root, _, movie = recording(run, stem)
    labels = declared_source(root, movie, "labels")
    masks = plane(labels, frame)
    if masks.shape != (result["height"], result["width"]):
        raise ValueError("Saved raw and mask planes do not share the same field dimensions.")
    layers = {"accepted": geometry(masks)}
    from .measurement_geometry import layers as measurement_layers
    measured = measurement_layers(root, movie, masks, frame)
    if measured is not None:
        return {**result, 'layers':measured, 'mask_label':'Original outline, measured aperture and local background'}
    base = movie.get("display", {}).get("base_labels")
    if base:
        path = Path(base); path = path if path.is_absolute() else root / path
        previous = plane(path.resolve(), frame)
        if previous.shape != masks.shape:
            raise ValueError("Declared comparison masks have different field dimensions.")
        layers["base"] = geometry(previous)
    return {**result, "layers": layers, "mask_label": "Saved measurement masks"}


def load_frame_image(run, stem, frame, *, preview=False):
    """Exact source pixels without contour extraction, for bounded browser preloading."""
    root, _, movie = recording(run, stem)
    if frame < 0:
        raise ValueError("Frame must be nonnegative.")
    raw = declared_source(root, movie, "raw")
    source_frame = frame + int(movie.get("spec", {}).get("source_frame_offset", 0))
    pixels = plane(raw, source_frame)
    contrast = movie.get("display", {}).get("contrast") or _contrast(*signature(raw))
    if len(contrast) != 2 or not all(np.isfinite(contrast)) or contrast[1] <= contrast[0]:
        raise ValueError("The declared display contrast is invalid.")
    reduced = preview and max(pixels.shape) > 256
    return {"stem": stem, "frame_index": frame, "source_frame_index": source_frame,
            "width": pixels.shape[1], "height": pixels.shape[0], "image": png(pixels, contrast, preview=reduced),
            **({"preview": True, "display_only": True} if reduced else {}),
            "contrast": list(contrast)}


def load_history(run, stem, identity, frame=0):
    root, _, movie = recording(run, stem)
    labels = declared_source(root, movie, "labels")
    with tifffile.TiffFile(signature(labels)[0]) as tif:
        shape = tif.series[0].shape
        count = shape[0] if len(shape) == 3 else 1
    if not 0 <= frame < count:
        raise ValueError("Frame is outside the saved mask recording.")
    stride = max(1, int(np.ceil(count / MAX_HISTORY)))
    exact = count <= MAX_EXACT_HISTORY
    indices = range(count) if exact else sorted(set(range(0, count, stride)) | {frame, count - 1})
    # Keep one TIFF handle open while reading history; opening it once per frame
    # dominates the first linked view for ordinary multi-page recordings.
    with tifffile.TiffFile(signature(labels)[0]) as tif:
        series = tif.series[0]
        if len(series.pages) == count:
            observed = [{"frame_index": i,
                         "observation": next(iter(geometry(series.pages[i].asarray(), identity)), None)}
                        for i in indices]
        else:
            observed = [{"frame_index": i, "observation": next(iter(geometry(plane(labels, i), identity)), None)}
                        for i in indices]
    return {"stem": stem, "identity": identity, "count": count, "stride": stride, "exact": exact,
            "observations": observed,
            "message": (f"Exact saved outline for every frame; background shows every {stride} frame(s)."
                        if exact else f"Background outlines sampled every {stride} frame(s); selected frame loads separately.")}


def load_cell_images(run, stem, identity):
    """Preload native-pixel crops for one saved identity within a fixed memory budget."""
    root, _, movie = recording(run, stem)
    from .measurement_geometry import parameters
    if parameters(movie) is not None:
        # Fetch one exact aperture/ring with its native crop, rather than
        # caching a strip that only contains the original outline layer.
        return {'stem':stem, 'identity':identity, 'frames':[], 'available':False}
    labels = declared_source(root, movie, "labels")
    raw = declared_source(root, movie, "raw")
    with _plane_reader(labels) as label_frame, _plane_reader(raw) as raw_frame:
        return _cell_images(movie, identity, labels, raw, label_frame, raw_frame)


def _cell_images(movie, identity, labels, raw, label_frame, raw_frame):
    stem = movie["stem"]
    with tifffile.TiffFile(signature(labels)[0]) as tif:
        shape = tif.series[0].shape
        count = shape[0] if len(shape) == 3 else 1
    if count > MAX_EXACT_HISTORY:
        return {"stem": stem, "identity": identity, "frames": [], "available": False}
    bounds = []
    for frame in range(count):
        yy, xx = np.nonzero(label_frame(frame) == identity)
        bounds.append((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1) if yy.size else None)
    observed = [i for i, b in enumerate(bounds) if b is not None]
    if not observed:
        return {"stem": stem, "identity": identity, "frames": [], "available": False}
    nearest = [bounds[min(observed, key=lambda i: abs(i - frame))] for frame in range(count)]
    height, width = label_frame(0).shape
    crops = [_padded_bounds(_union_bounds(nearest[frame], bounds[frame - 1] if frame else None), width, height)
             for frame in range(count)]
    if sum((x1 - x0) * (y1 - y0) * 4 for x0, y0, x1, y1 in crops) > MAX_CELL_STRIP_BYTES:
        return {"stem": stem, "identity": identity, "frames": [], "available": False}
    contrast = movie.get("display", {}).get("contrast") or _contrast(*signature(raw))
    if len(contrast) != 2 or not all(np.isfinite(contrast)) or contrast[1] <= contrast[0]:
        raise ValueError("The declared display contrast is invalid.")
    offset = int(movie.get("spec", {}).get("source_frame_offset", 0))
    frames = []
    for frame, crop in enumerate(crops):
        pixels = raw_frame(frame + offset)
        if pixels.shape != (height, width):
            raise ValueError("Saved raw and mask planes do not share the same field dimensions.")
        x0, y0, x1, y1 = crop
        current = label_frame(frame)
        previous = label_frame(frame - 1) if frame else None
        frames.append({"frame_index": frame, "source_frame_index": frame + offset,
                       "width": width, "height": height, "image_bounds": list(crop),
                       "image": png(pixels[y0:y1, x0:x1], contrast), "contrast": list(contrast),
                       "change": _change_image(current, previous, identity, crop), "change_bounds": list(crop),
                       "observation": next(iter(geometry(current, identity)), None)})
    return {"stem": stem, "identity": identity, "frames": frames, "available": True}


def load_cell_frame(run, stem, identity, frame):
    """Current native crop, signed changes and all saved-mask cell centres."""
    root, _, movie = recording(run, stem)
    labels = declared_source(root, movie, "labels")
    current = plane(labels, frame)
    if current.dtype.kind not in "iu" or np.any(current < 0):
        raise ValueError("Saved masks must contain nonnegative integer identities.")
    cy, cx = np.nonzero(current)
    identities, indices = np.unique(current[cy, cx], return_inverse=True)
    if len(identities) > 1000:
        raise ValueError("This plane exceeds the 1,000-cell display budget.")
    counts = np.bincount(indices)
    centres_x = np.bincount(indices, weights=cx) / counts + .5
    centres_y = np.bincount(indices, weights=cy) / counts + .5
    positions = [{"identity": int(i), "x": float(x), "y": float(y)}
                 for i, x, y in zip(identities, centres_x, centres_y)]
    previous = plane(labels, frame - 1) if frame else None
    yy, xx = np.nonzero((current == identity) | ((previous == identity) if previous is not None else False))
    if not yy.size:
        return {"frame_index": frame, "identity": identity, "available": False,
                  "change": None, "change_bounds": None, "positions": positions}
    height, width = current.shape
    from .measurement_geometry import parameters as measurement_parameters
    measured_options = measurement_parameters(movie)
    padding = max(15, int(np.ceil(measured_options['margin_px']+measured_options['background_outer_px']))+2) if measured_options else 15
    crop = _padded_bounds((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1), width, height, padding)
    raw = declared_source(root, movie, "raw")
    offset = int(movie.get("spec", {}).get("source_frame_offset", 0))
    pixels = plane(raw, frame + offset)
    if pixels.shape != current.shape:
        raise ValueError("Saved raw and mask planes do not share the same field dimensions.")
    contrast = movie.get("display", {}).get("contrast") or _contrast(*signature(raw))
    if len(contrast) != 2 or not all(np.isfinite(contrast)) or contrast[1] <= contrast[0]:
        raise ValueError("The declared display contrast is invalid.")
    x0, y0, x1, y1 = crop
    from .measurement_geometry import layers as measurement_layers
    measured = measurement_layers(root, movie, current, frame, identity)
    return {"frame_index": frame, "source_frame_index": frame + offset, "identity": identity,
            "width": width, "height": height, "image_bounds": list(crop),
            "image": png(pixels[y0:y1, x0:x1], contrast), "contrast": list(contrast),
              "observation": next(iter(geometry(current, identity)), None), "available": True,
              "positions": positions, **({'layers':measured} if measured is not None else {}),
            "change": _change_image(current, previous, identity, crop), "change_bounds": list(crop)}
