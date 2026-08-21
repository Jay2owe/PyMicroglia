"""Timestamps, captions and the even-dimension padding an encoder insists on.

The elapsed-time wording is shared across four exporters and reads
``Time: 3 h 30 min``. It is kept in one function because a movie whose caption
wording changed between two recordings looks like two different experiments,
and because the wording appears verbatim in a manifest column that has already
been read into a methods paragraph.

Two arrangements, both in use and both kept:

* a **band** above the image — black, 48 px, text centred. It can be cropped
  off later without re-rendering, and every exporter that timestamps a
  publication movie uses it.
* a **caption** burnt into the top-left corner, white with a black outline and
  no band, so the frame keeps its native size. The tuning review videos use
  this so several can be watched side by side without the bands stacking up.

Fonts are looked up by file, in order, first hit winning, and fall back to
Pillow's built-in. A missing Arial changes the glyphs, not the geometry: the
band height and the text position are computed from the drawn text's own
bounding box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "DEFAULT_FONTS",
    "elapsed_label",
    "font_for_band",
    "timestamp_band",
    "clock_label",
    "caption_band",
    "corner_label",
    "pad_to_even",
    "pad_into",
]

#: Tried in order, first that exists winning. Windows paths because that is
#: where these recordings are processed; a machine without them gets Pillow's
#: built-in font and a movie that still says the right time.
DEFAULT_FONTS: tuple[str, ...] = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

#: Band height times this is the point size. 0.54 of 48 px is 25 pt, which is
#: what every timestamped movie in the project was rendered at.
FONT_HEIGHT_FRACTION = 0.54
FONT_MIN_SIZE = 14


def elapsed_label(frame_index: int, interval_h: float) -> str:
    """``Time: 3 h 30 min``. The house wording, shared by four exporters.

    Minutes are rounded once, from the total, rather than accumulated per
    frame: rounding each frame's contribution makes the last label of a
    thousand-frame recording drift by minutes.
    """
    total_minutes = int(round(float(frame_index) * float(interval_h) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"Time: {hours} h {minutes:02d} min"


def font_for_band(band_height: int, candidates: Sequence[str] = DEFAULT_FONTS,
                  *, fallback_size: int | None = None):
    from PIL import ImageFont

    size = fallback_size or max(FONT_MIN_SIZE,
                                int(round(band_height * FONT_HEIGHT_FRACTION)))
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                continue
    return ImageFont.load_default()


def timestamp_band(rgb, *, frame_index: int, interval_h: float,
                   band_height: int = 48,
                   fonts: Sequence[str] = DEFAULT_FONTS,
                   font_size: int | None = None) -> Any:
    """A black band above the image carrying elapsed experimental time.

    Returns the taller image. Nothing is drawn over the data — that is the
    point of a band rather than an overlay, and it is why the band can be
    cropped off afterwards with one ffmpeg call instead of a re-render.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if band_height <= 0:
        return rgb
    frame = np.asarray(rgb, np.uint8)
    canvas = np.zeros((frame.shape[0] + int(band_height), frame.shape[1], 3),
                      np.uint8)
    canvas[int(band_height):] = frame
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    label = elapsed_label(frame_index, interval_h)
    font = font_for_band(int(band_height), fonts, fallback_size=font_size)
    bounds = draw.textbbox((0, 0), label, font=font)
    x = max(0, (canvas.shape[1] - (bounds[2] - bounds[0])) // 2)
    y = max(0, (int(band_height) - (bounds[3] - bounds[1])) // 2 - bounds[1])
    draw.text((x, y), label, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def corner_label(rgb, text: str, *, position: tuple[int, int] = (8, 8)) -> Any:
    """A caption in the top-left, white with a black outline, no band."""
    import numpy as np
    from PIL import Image, ImageDraw

    if not text:
        return rgb
    image = Image.fromarray(np.asarray(rgb, np.uint8).copy())
    ImageDraw.Draw(image).text(position, text, fill="white", stroke_width=2,
                               stroke_fill="black")
    return np.asarray(image)


def pad_to_even(rgb) -> Any:
    """``yuv420p`` halves both dimensions, so both have to be even."""
    import numpy as np

    frame = np.asarray(rgb)
    pad_y = frame.shape[0] % 2
    pad_x = frame.shape[1] % 2
    if not (pad_y or pad_x):
        return frame
    return np.pad(frame, ((0, pad_y), (0, pad_x), (0, 0)))


def pad_into(rgb, height: int, width: int) -> Any:
    """One frame placed top-left in a fixed canvas.

    Two of the engines size the canvas once, before the loop, and paste every
    frame into it. Same result as :func:`pad_to_even` when the frames are all
    one size, and it keeps the stream's geometry constant when they are not —
    which an encoder fed a raw pipe requires.
    """
    import numpy as np

    frame = np.asarray(rgb, np.uint8)
    canvas = np.zeros((int(height), int(width), 3), np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    return canvas


# --------------------------------------------------- the second band variant
def clock_label(frame_index: int, interval_seconds: float) -> str:
    """``Time: 00:30:00``. The organotypic exporter's wording.

    A second wording, kept rather than unified, because it appears in a
    published methods paragraph — "the 1,018 frames are timestamped at 30
    seconds per frame (00:00:00 through ...)" — and because at 30 seconds a
    frame the ``h``/``min`` form would round every caption to the same minute.
    """
    seconds = int(round(float(frame_index) * float(interval_seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"


def caption_band(rgb, label: str, *, band_height: int = 48,
                 legend: str = "", left_title: str = "", right_title: str = "",
                 font_directory: str = "C:/Windows/Fonts",
                 font_regular: str = "arial.ttf",
                 font_bold: str = "arialbd.ttf",
                 timestamp_size: int = 23, legend_size: int = 16,
                 title_size: int = 13) -> Any:
    """A band carrying a left-aligned time, a right-aligned legend and titles.

    The organotypic exporter's arrangement. The legend names which colour is
    which reporter, which matters on a two-colour composite where nothing else
    on screen says so; the two titles label the halves of a side-by-side panel.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    frame = np.asarray(rgb, np.uint8)
    canvas = np.zeros((frame.shape[0] + int(band_height), frame.shape[1], 3),
                      np.uint8)
    canvas[int(band_height):] = frame
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    def face(size: int, *, bold: bool = False):
        name = font_bold if bold else font_regular
        return font_for_band(int(band_height),
                             (str(Path(font_directory) / name),),
                             fallback_size=size)

    draw.text((12, 11), label, fill=(255, 255, 255),
              font=face(timestamp_size, bold=True))
    if legend:
        legend_font = face(legend_size)
        box = draw.textbbox((0, 0), legend, font=legend_font)
        draw.text((frame.shape[1] - (box[2] - box[0]) - 12, 15), legend,
                  fill=(230, 230, 230), font=legend_font)
    if left_title and right_title:
        title_font = face(title_size)
        draw.text((12, 30), left_title, fill=(190, 190, 190), font=title_font)
        draw.text((frame.shape[1] // 2 + 12, 30), right_title,
                  fill=(190, 190, 190), font=title_font)
    return np.asarray(image)
