"""The frozen CD68 three-frame detector, as a shape-only proposal provider.

Inference port of Motion ISSUE-006 m1/m2, R02 A004 model and A006 decoding,
whose foreground was frozen in ISSUE-007. No tracking or training code runs.
Weights remain an external experimental artifact. The saved preprocessing
includes float16 quantisation before inference; omitting it changes predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

OPTIONS = {"contrast_percentiles", "gamma", "foreground_threshold",
           "seed_threshold", "min_seed_px", "min_object_px"}


def build_model(base=16, depth=3):
    """Build the exact three-input, three-class architecture in the checkpoint."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ConvBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        def forward(self, value):
            return self.net(value)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [base * 2 ** i for i in range(depth + 1)]
            self.encoders = nn.ModuleList()
            channels = 3
            for width in widths[:-1]:
                self.encoders.append(ConvBlock(channels, width))
                channels = width
            self.bottleneck = ConvBlock(channels, widths[-1])
            channels = widths[-1]
            self.decoders = nn.ModuleList()
            for width in reversed(widths[:-1]):
                self.decoders.append(ConvBlock(channels + width, width))
                channels = width
            self.head = nn.Conv2d(channels, 3, 1)

        def forward(self, value):
            skips = []
            for block in self.encoders:
                value = block(value)
                skips.append(value)
                value = F.max_pool2d(value, 2)
            value = self.bottleneck(value)
            for block, skip in zip(self.decoders, reversed(skips)):
                value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
                value = block(torch.cat([value, skip], dim=1))
            return self.head(value)

    return UNet()


def normalise(raw, *, contrast_percentiles=(0.5, 99.8), gamma=0.5, output=None):
    percentiles = np.asarray(contrast_percentiles, float)
    if percentiles.shape != (2,) or not 0 <= percentiles[0] < percentiles[1] <= 100 \
            or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("invalid CD68 contrast percentiles or gamma")
    low, high = (float(v) for v in np.percentile(raw[::5, ::3, ::3], percentiles))
    if output is None:
        unit = np.clip((raw.astype(np.float32) - low) / max(high - low, 1e-6), 0, 4)
        images = np.power(unit, gamma, dtype=np.float32).astype(np.float16)
    else:
        if output.shape!=raw.shape or output.dtype!=np.float16:raise ValueError('CD68 normalization output must match the movie shape and float16 dtype')
        images=output
        for t,frame in enumerate(raw):
            unit=np.clip((frame.astype(np.float32)-low)/max(high-low,1e-6),0,4)
            images[t]=np.power(unit,gamma,dtype=np.float32).astype(np.float16)
            if (t+1)%16==0 and hasattr(images,'flush'):images.flush()
    return images, {"contrast_percentiles": list(percentiles), "gamma": gamma,
                    "contrast_low": low, "contrast_high": high,
                    "clip_max": 4.0, "storage_quantisation": "float16"}


def decode(probabilities, *, seed_threshold=0.3, foreground_threshold=0.5,
           min_seed_px=4, min_object_px=12):
    """Reproduce the accepted foreground decoding; discard its object names."""
    from skimage.segmentation import watershed

    interior = probabilities[1]
    foreground = (interior + probabilities[2]) > foreground_threshold
    seeds, _ = ndi.label(interior > seed_threshold)
    if min_seed_px > 1 and seeds.max():
        sizes = np.bincount(seeds.ravel())
        seeds[np.isin(seeds, np.flatnonzero(sizes < min_seed_px))] = 0
        seeds, _ = ndi.label(seeds > 0)
    if not seeds.max():
        return np.zeros(interior.shape, bool)
    labels = watershed(-interior, markers=seeds, mask=foreground)
    rest, _ = ndi.label(foreground & (labels == 0))
    rest[rest > 0] += int(labels.max())
    labels += rest
    if min_object_px > 1 and labels.max():
        sizes = np.bincount(labels.ravel())
        labels[np.isin(labels, np.flatnonzero(sizes < min_object_px))] = 0
    return labels > 0


def exported_probabilities(probabilities):
    """A006 decoded the saved uint8 maps, including truncation to 255 steps."""
    return np.clip(probabilities * 255, 0, 255).astype(np.uint8).astype(np.float32) / 255.0


def predict(raw, *, weights, threads=4, options=None, memory_format="contiguous",
            progress=None, workspace=None,output=None):
    """Produce the frozen foreground with optional file-backed context and output.

    ``output`` is a caller-owned writable boolean movie. The same decoding
    fills it frame by frame; the caller keeps its mapping open until finished.
    Omitting it retains the established in-memory result and metadata.
    """
    if workspace is None:
        return _predict(raw,weights=weights,threads=threads,options=options,memory_format=memory_format,progress=progress,foreground_output=output)
    import tempfile
    folder=Path(workspace);folder.mkdir(parents=True,exist_ok=True)
    raw=np.asarray(raw)
    if raw.ndim!=3 or not raw.size:raise ValueError('CD68 images require a nonempty time, height, width stack')
    with tempfile.TemporaryDirectory(prefix='cd68-context-',dir=folder) as scratch:
        context=np.memmap(Path(scratch)/'images.dat',mode='w+',dtype=np.float16,shape=raw.shape)
        try:
            return _predict(raw,weights=weights,threads=threads,options=options,memory_format=memory_format,
                progress=progress,normalise_output=context,foreground_output=output)
        finally:context._mmap.close()


def _predict(raw, *, weights, threads=4, options=None, memory_format="contiguous",
             progress=None,normalise_output=None,foreground_output=None):
    """Produce cell foreground from raw fluorescence with adjacent-frame context."""
    import torch
    from torch.nn import functional as F

    options = dict(options or {})
    if memory_format not in {"contiguous", "channels_last"}:
        raise ValueError("memory_format must be contiguous or channels_last")
    if set(options) - OPTIONS:
        raise ValueError(f"unknown CD68 options: {sorted(set(options) - OPTIONS)}")
    if weights is None:
        raise ValueError("cd68 refinement requires the trained CD68 checkpoint in weights")
    checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=True)
    params = checkpoint["params"]
    if int(params.get("context_frames", -1)) != 1 or any(params.get(k) for k in (
            "use_motion", "use_prior", "centre_head", "carry_seeds")):
        raise ValueError("CD68 adapter requires the three-frame foreground-only checkpoint")
    raw = np.asarray(raw)
    if raw.ndim != 3 or not len(raw) or not all(np.isfinite(frame).all() for frame in raw):
        raise ValueError("CD68 images must be a finite time, height, width stack")
    if foreground_output is not None:
        if foreground_output.shape!=raw.shape or foreground_output.dtype!=bool or not foreground_output.flags.writeable:
            raise ValueError('CD68 foreground output must be a writable bool array matching the movie')
        if np.shares_memory(foreground_output,raw):raise ValueError('CD68 foreground output must not overwrite source pixels')
    images, scaling = normalise(raw, output=normalise_output, **{k: v for k, v in options.items()
                                      if k in {"contrast_percentiles", "gamma"}})
    decoding = {"seed_threshold": 0.3, "foreground_threshold": 0.5,
                "min_seed_px": 4, "min_object_px": 12}
    decoding.update({k: v for k, v in options.items() if k in decoding})
    for key in ("seed_threshold", "foreground_threshold"):
        if not 0 <= decoding[key] <= 1:
            raise ValueError(f"{key} must be in 0..1")
    for key in ("min_seed_px", "min_object_px"):
        if not isinstance(decoding[key], int) or decoding[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    previous_threads = torch.get_num_threads()
    try:
        if threads > 0:
            torch.set_num_threads(threads)
        depth = int(params["depth"])
        model = build_model(int(params["base_channels"]), depth)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        if memory_format == "channels_last":
            model = model.to(memory_format=torch.channels_last)
        height, width = raw.shape[1:]
        pad = (0, (-width) % (2 ** depth), 0, (-height) % (2 ** depth))
        foreground = np.zeros(raw.shape, bool) if foreground_output is None else foreground_output
        with torch.no_grad():
            for t in range(len(images)):
                context = np.stack([images[min(max(t + d, 0), len(images) - 1)]
                                    for d in (-1, 0, 1)]).astype(np.float32)
                tensor = F.pad(torch.from_numpy(context)[None], pad, mode="reflect")
                if memory_format == "channels_last":
                    tensor = tensor.to(memory_format=torch.channels_last)
                probabilities = F.softmax(model(tensor), dim=1)[0, :, :height, :width].numpy()
                foreground[t] = decode(exported_probabilities(probabilities), **decoding)
                if (t+1)%16==0 and hasattr(foreground,'flush'):foreground.flush()
                if progress is not None:
                    progress(t + 1, len(images))
    finally:
        torch.set_num_threads(previous_threads)
    return {"kind": "binary", "data": foreground,
            "metadata": {"model": "cd68_three_frame_unet", "context_frames": 1,
                         "preprocessing": scaling, "decoding": decoding,
                         "probability_quantisation": "uint8_truncate_then_divide_by_255",
                         "identity_prediction": False}}
