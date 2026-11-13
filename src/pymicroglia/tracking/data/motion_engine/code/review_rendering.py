"""Established full-field issue-review rendering style.

This is the production-neutral home of the visual conventions previously shared by
loading the Issue 008 renderer: non-repeating identity colours, component-centred
labels, fixed whole-movie raw contrast, and a 24-pixel header.
"""
from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from common import display_raw, label_edges
from pipeline import _motion_rgb


def identity_colour(identity: int) -> np.ndarray:
    hue = (identity * 0.618033988749895) % 1.0
    saturation = 0.62 + 0.08 * (identity % 4)
    value = 0.82 + 0.08 * (identity % 3)
    rgb = colorsys.hsv_to_rgb(hue, min(saturation, 0.9), min(value, 0.98))
    return np.array([round(channel * 255) for channel in rgb], np.uint8)


def identity_annotations(labels: np.ndarray) -> list[list[tuple]]:
    """Locate label text once so multiple panels can reuse the same positions."""
    frames = []
    for t in range(len(labels)):
        commands = []
        for identity, bounds in enumerate(ndi.find_objects(labels[t]), start=1):
            if bounds is None:
                continue
            parts, count = ndi.label(labels[t][bounds] == identity,
                                     structure=np.ones((3, 3), np.uint8))
            for component in range(1, count + 1):
                mask = parts == component
                if int(mask.sum()) < 8:
                    continue
                y, x = ndi.center_of_mass(mask)
                y += bounds[0].start
                x += bounds[1].start
                colour = tuple(map(int, identity_colour(identity)))
                commands.append(((round(x), round(y)), str(identity), colour))
        frames.append(commands)
    return frames


def annotate_identities(stack: np.ndarray, labels: np.ndarray,
                        *, annotations: list[list[tuple]] | None = None
                        ) -> np.ndarray:
    """Draw every substantial connected component exactly as prior reviews did."""
    result = stack.copy()
    commands_by_frame = (identity_annotations(labels) if annotations is None
                         else annotations)
    if len(commands_by_frame) != len(result):
        raise ValueError("identity annotations must match review frames")
    for t, commands in enumerate(commands_by_frame):
        image = Image.fromarray(result[t])
        draw = ImageDraw.Draw(image)
        for xy, label, colour in commands:
            draw.text(xy, label, fill=colour,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        result[t] = np.asarray(image)
    return result


def _trail_centres(labels: np.ndarray) -> list[dict[int, tuple[int, int]]]:
    """Use the largest observed component of each identity in each frame."""
    frames = []
    for frame in labels:
        centres = {}
        for identity, bounds in enumerate(ndi.find_objects(frame), start=1):
            if bounds is None:
                continue
            parts, count = ndi.label(frame[bounds] == identity,
                                     structure=np.ones((3, 3), np.uint8))
            if not count:
                continue
            sizes = np.bincount(parts.ravel())
            sizes[0] = 0
            part = int(np.argmax(sizes))
            y, x = ndi.center_of_mass(parts == part)
            centres[identity] = (round(x + bounds[1].start),
                                 round(y + bounds[0].start))
        frames.append(centres)
    return frames


def _paint_motion_trails(stack: np.ndarray, labels: np.ndarray, *,
                         trail_frames: int, trail_opacity: float) -> np.ndarray:
    """Draw earlier positions dimmer, only across consecutive observed masks."""
    centres = _trail_centres(labels)
    result = stack.copy()
    height, width = result.shape[1:3]
    for frame in range(1, len(result)):
        active = centres[frame]
        if not active:
            continue
        layer = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(layer)
        for identity in active:
            colour = tuple(map(int, identity_colour(identity)))
            segments = []
            for age in range(1, min(frame, trail_frames) + 1):
                newer = centres[frame - age + 1].get(identity)
                older = centres[frame - age].get(identity)
                if newer is None or older is None:
                    break
                segments.append((age, older, newer))
            for age, older, newer in reversed(segments):
                alpha = round(255 * trail_opacity * (trail_frames - age + 1)
                              / trail_frames)
                draw.line((older, newer), fill=(*colour, alpha), width=2)
        if layer.getbbox() is not None:
            painted = np.array(Image.alpha_composite(
                Image.fromarray(result[frame]).convert("RGBA"), layer).convert("RGB"))
            painted[labels[frame] > 0] = result[frame][labels[frame] > 0]
            result[frame] = painted
    return result


def unique_outline(raw: np.ndarray, labels: np.ndarray,
                   motion: np.ndarray | None = None,
                   *, raw_display: np.ndarray | None = None,
                   annotations: list[list[tuple]] | None = None,
                   edge_mask: np.ndarray | None = None,
                   show_motion_trails: bool = False,
                   trail_frames: int = 4,
                   trail_opacity: float = 0.65) -> np.ndarray:
    if not isinstance(show_motion_trails, bool):
        raise ValueError("show_motion_trails must be true or false")
    if isinstance(trail_frames, bool) or not isinstance(trail_frames, int) or trail_frames < 1:
        raise ValueError("trail_frames must be a positive whole number")
    if not np.isfinite(trail_opacity) or not 0 <= trail_opacity <= 1:
        raise ValueError("trail_opacity must be between zero and one")
    out = ((display_raw(raw) if raw_display is None else raw_display.copy())
           if motion is None else _motion_rgb(raw, motion, labels, raw_display=raw_display))
    if show_motion_trails and trail_opacity > 0:
        out = _paint_motion_trails(out, labels, trail_frames=trail_frames,
                                   trail_opacity=trail_opacity)
    edge = label_edges(labels, thick=2) if edge_mask is None else edge_mask
    identities = np.unique(labels)
    palette = np.zeros((int(identities[-1]) + 1, 3), np.uint8)
    for identity in identities:
        if identity > 0:
            palette[int(identity)] = identity_colour(int(identity))
    positive_edge = edge & (labels > 0)
    out[positive_edge] = palette[labels[positive_edge]]
    return annotate_identities(out, labels, annotations=annotations)


def header(stack: np.ndarray, title: str, source_offset: int,
           notes: list[str] | None = None, height: int = 24) -> np.ndarray:
    result = np.zeros((len(stack), stack.shape[1] + height,
                       stack.shape[2], 3), np.uint8)
    result[:, height:] = stack
    for frame in range(len(result)):
        image = Image.fromarray(result[frame])
        line = (f"{title} | review {frame + 1} | "
                f"source ImageJ {frame + source_offset + 1}")
        if notes and notes[frame]:
            line += f" | {notes[frame]}"
        ImageDraw.Draw(image).text((4, 4), line, fill=(255, 255, 255))
        result[frame] = np.asarray(image)
    return result
