"""Resolve saved label/photon inputs and their recorded native-frame alignment."""
from ..measure.mask_io import stack, labels, clock, paired


def load(source, *, need_raw=True):
    label_path = source.input_path("labels")
    if label_path is None:
        raise ValueError("the saved run has no recorded label stack")
    masks = labels(label_path)
    spec = source.movie.get("spec", {})
    offset, cadence = clock(spec.get("source_frame_offset", 0), source.interval / 60)
    raw_path = source.input_path("raw") if need_raw else None
    if need_raw and raw_path is None:
        raise ValueError("the saved run has no original photon stack")
    raw = stack(raw_path) if need_raw else None
    if need_raw:
        raw, masks = paired(raw, masks, offset)
    return {"labels": masks, "raw": raw, "offset": offset, "cadence": cadence,
            "label_path": label_path, "raw_path": raw_path}
