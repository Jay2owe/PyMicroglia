"""The encoder, the frame rate, and the contract people actually think in.

Playback speed is given in **experimental hours per second**, not in frames per
second. That is the number a reader of a movie needs: at 12, one biological day
takes two seconds of screen time whatever the acquisition interval happened to
be. The frame rate falls out of it, and no frame is ever dropped or duplicated.

    fps = hours_per_second / frame_interval_h

Encoding goes through ``imageio-ffmpeg`` rather than a subprocess. Three of the
four engines shelled out to ffmpeg directly; this package may not, and a test
enforces that — a package that shells out is a package whose behaviour depends
on a PATH nobody recorded. ``imageio-ffmpeg`` ships its own binary, so
``pymicroglia doctor`` can say whether video export will work before a six-hour
run rather than after it.

**The x264 settings are per engine, not shared.** They genuinely differ, and
each was chosen against a named recording:

===================== ======== ============================================
profile               preset   why
===================== ======== ============================================
``grainy_display``    slow     ``-tune grain``. Low-light bioluminescence is
                               noise as much as signal, and tuning for grain
                               is what stops the noise field smearing into
                               blocks and taking dim processes with it.
``standard``          slow     The two-channel exporters. No grain tuning:
                               their footage is brighter and cleaner.
``fast``              medium   ``phase_green_red_video_export.py``, which
                               renders five videos of 1,018 frames each and
                               would otherwise take most of an evening.
===================== ======== ============================================

Reproducing an engine's video means reproducing its profile. Which one a movie
was made with is written into its report, because a movie re-encoded under a
different preset is a different file that decodes to almost the same frames,
and "almost" is a thing a reader should be told rather than left to discover.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "PROFILES",
    "DEFAULT_MIN_FPS",
    "DEFAULT_MAX_FPS",
    "frame_rate",
    "check_frame_rate",
    "write_video",
    "available",
    "ffmpeg_version",
]

#: Refuse outside this range. A rate no display can show is a wrong
#: ``hours_per_second``, not a fast movie; under 1 fps the result is a
#: slideshow many players stumble on.
DEFAULT_MIN_FPS = 1.0
DEFAULT_MAX_FPS = 60.0

PROFILES: dict[str, dict[str, Any]] = {
    "grainy_display": {
        "preset": "slow",
        "extra": ["-tune", "grain", "-profile:v", "high", "-level", "4.0"],
        "crf": 14,
        "used_by": "tiff_stack_to_mp4",
    },
    "standard": {
        "preset": "slow",
        "extra": [],
        "crf": 18,
        "used_by": "red_only_video, composite_video",
    },
    "fast": {
        "preset": "medium",
        "extra": [],
        "crf": 18,
        "used_by": "phase_green_red_video",
    },
}


def frame_rate(frame_interval_h: float, hours_per_second: float) -> float:
    """Frames per second covering the requested experimental hours per second.

    This is the control people reason about; fps is derived and never set
    directly. Halving ``hours_per_second`` halves the frame rate and doubles
    the run time — it does not resample.
    """
    interval = float(frame_interval_h)
    if interval <= 0:
        raise ValueError("frame_interval_h must be positive; the movie's "
                         "elapsed time is meaningless without it")
    return float(hours_per_second) / interval


def check_frame_rate(fps: float, *, minimum: float = DEFAULT_MIN_FPS,
                     maximum: float = DEFAULT_MAX_FPS, what: str = "") -> float:
    if not minimum <= fps <= maximum:
        raise ValueError(
            f"{what or 'this recording'} asks for {fps:.2f} fps, outside "
            f"{minimum:g}..{maximum:g}. Change hours_per_second — a rate no "
            f"display can show is a wrong speed, not a fast movie.")
    return float(fps)


def available() -> bool:
    """Whether an ffmpeg this package can drive is present."""
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return False
    try:
        import imageio_ffmpeg

        return bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return False


def ffmpeg_version() -> str | None:
    """The bundled ffmpeg's version, for the run record.

    Recorded because ffmpeg output is not bit-reproducible across versions.
    Two runs that differ only here produce different bytes and nearly identical
    frames, and a reader comparing file hashes should know that before they
    start.
    """
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_version())
    except Exception:
        return None


def write_video(path, frames: Iterable[Any], *, fps: float,
                profile: str = "standard", crf: int | None = None,
                overwrite: bool = False) -> Path:
    """Encode a stream of RGB frames to H.264 in an MP4.

    Written to a temporary name in the same folder and moved into place, so an
    interrupted encode leaves no half-written movie that looks finished.
    """
    import imageio.v2 as imageio

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {target}. Pass overwrite=True to replace "
            f"it — a re-run silently replacing the movie you were comparing "
            f"against is a mistake that only has to happen once.")
    if profile not in PROFILES:
        raise ValueError(f"unknown encoder profile {profile!r}; "
                         f"choose one of {sorted(PROFILES)}")

    settings = PROFILES[profile]
    quality = int(settings["crf"] if crf is None else crf)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp.mp4")

    writer = imageio.get_writer(
        str(temporary), fps=float(fps), codec="libx264", quality=None,
        macro_block_size=1, pixelformat="yuv420p",
        ffmpeg_params=["-crf", str(quality), *settings["extra"],
                       "-preset", str(settings["preset"]),
                       "-movflags", "+faststart"],
    )
    count = 0
    try:
        for frame in frames:
            writer.append_data(frame)
            count += 1
    finally:
        writer.close()

    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("no frames were written; refusing to leave an empty "
                         "MP4 that looks like a finished movie")
    temporary.replace(target)
    return target
