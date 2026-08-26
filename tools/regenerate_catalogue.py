#!/usr/bin/env python
"""Regenerate the action catalogue PyMicroglia ships.

Run this when a source protocol's parameter block changes and you want the
package to pick the change up, or when adding an action.

    python tools/regenerate_catalogue.py                    # default location
    python tools/regenerate_catalogue.py --protocols <dir>  # somewhere else
    python tools/regenerate_catalogue.py --check            # fail if stale

Why a generated file rather than reading the scripts at run time: PyMicroglia
depends on nothing in ``Protocols/Analysis``. The catalogue is seeded from those
scripts once, here, and then belongs to this package — free to be edited,
renamed and extended without touching them, and present whether or not they are.

This script is maintenance tooling. It is not importable from the package and
does not ship with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pymicroglia import params, scn_outline  # noqa: E402

DEFAULT_PROTOCOLS = REPO.parent / "Protocols"
OUTPUT = REPO / "src" / "pymicroglia" / "data" / "actions.json"

#: Self-contained actions may advance after their source protocol is frozen.
#: Read their live constant rather than copying it here, so the run-record key
#: cannot lag behind the method that actually produced the pixels.
METHOD_VERSION_OVERRIDES = {
    "automatic_scn_outline": scn_outline.METHOD_VERSION,
}

# The outline moved to PySCNSlice, so ``scn_outline.METHOD_VERSION`` is now read
# through a delegate and is the empty string when that package is not installed.
# Regenerating from such a machine would write an empty version into the
# catalogue — and ``recording._method_version`` reads the catalogue *first*, so
# every later run record would silently lose the field that decides whether two
# runs are comparable. Refuse instead: this tool runs on a developer's machine,
# where installing the extra is one command.
_EMPTY_OVERRIDES = sorted(
    name for name, version in METHOD_VERSION_OVERRIDES.items() if not version)
if _EMPTY_OVERRIDES:
    raise SystemExit(
        f"cannot regenerate the catalogue: {_EMPTY_OVERRIDES} report no method "
        'version. Install the packages that own them: pip install '
        '"PyMicroglia[scn]".')

#: Keys every action takes, which no PROTOCOL PARAMETERS block declares because
#: they are arguments rather than settings. The macro contract already requires
#: an analysis to accept the same four, so an agent that learns them once knows
#: them everywhere.
COMMON_PARAMS: list[dict] = [
    {"name": "source", "type": "path", "units": "-", "required": True, "default": None,
     "description": "What the action reads: the TIFF time series for a processing "
                    "or quality-control action, or the trace CSV for a trace panel. "
                    "The folder holding it also works."},
    {"name": "output_dir", "type": "path", "units": "-", "required": False, "default": None,
     "description": "Where results are written. Defaults to an AI_Exports folder "
                    "beside the source, which is where every protocol in this "
                    "project already puts them."},
    {"name": "output_name", "type": "str", "units": "-", "required": False, "default": None,
     "description": "Exact name for the primary output. Defaults to the source "
                    "stem plus the action's own suffix."},
    {"name": "overwrite", "type": "bool", "units": "-", "required": False, "default": False,
     "description": "Replace an existing output. Off by default: refusing to "
                    "overwrite is what stops a re-run destroying the figures you "
                    "were comparing against."},
]

#: Parameters an action takes that no PROTOCOL PARAMETERS block declares,
#: because they were command-line flags in the engine rather than settings.
#: Keyed by action name.
EXTRA_PARAMS: dict[str, list[dict]] = {
    "register": [
        {"name": "estimate_only", "type": "bool", "units": "-", "required": False,
         "default": False,
         "description": "Write the shifts and quality-control tables and no "
                        "registered TIFF. The tables are the analysis; the TIFF "
                        "is a 21 GB convenience that rebuilds from them."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored registration when one matches this exact "
                        "source, parameters and METHOD_VERSION. Turning it off "
                        "forces a fresh estimate, which reads the whole file."},
        {"name": "content_crop", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Crop to the largest tissue component plus a margin. "
                        "Off keeps the whole common valid field."},
    ],
    "register_three_channel": [
        {"name": "estimate_only", "type": "bool", "units": "-", "required": False,
         "default": False,
         "description": "Write the shifts and quality-control tables and no "
                        "registered TIFF."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored registration when one matches this exact "
                        "source, parameters and METHOD_VERSION."},
        {"name": "minimum_response", "type": "float", "units": "-",
         "required": False, "default": 0.20,
         "description": "Phase-correlation response below which the red "
                        "neuronal centroid is used instead. CAUTION: lowering "
                        "it accepts weaker correlations, and one bad step "
                        "carries into every later frame because the track "
                        "accumulates."},
    ],
    "export_registered_stack": [
        {"name": "shifts", "type": "path", "units": "-", "required": False,
         "default": None,
         "description": "A specific registration artefact to apply. Named "
                        "explicitly it always wins; left out, the one stored "
                        "registration matching this source is used, and two "
                        "matches is an error naming both."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Reserved for symmetry with the registration actions; "
                        "this action never estimates."},
        {"name": "compression_level", "type": "int", "units": "-",
         "required": False, "default": 4,
         "description": "TIFF zlib level, 0..9; 0 disables compression."},
    ],
    "remove_cosmic_rays": [
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored result when one matches this exact "
                        "source, parameters, METHOD_VERSION and upstream "
                        "registration. Turning it off forces a fresh pass over "
                        "every frame."},
    ],
    "remove_static_background": [
        {"name": "fluctuation_only", "type": "bool", "units": "-",
         "required": False, "default": False,
         "description": "Write the mean-subtracted stack instead, so the fixed "
                        "texture is genuinely subtracted rather than left below "
                        "the display floor, at the cost of a visibly noisier "
                        "background. Display only either way."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored result when one matches this exact "
                        "source, parameters, METHOD_VERSION and upstream "
                        "registration. Turning it off forces a fresh pass over "
                        "every frame."},
    ],
    "display_filter": [
        {"name": "injection_check_on", "type": "bool", "units": "-",
         "required": False, "default": False,
         "description": "Add an oscillation of known size to the recording, run "
                        "the filter twice and measure what comes back. The "
                        "filter's response depends on the data it is given, so "
                        "unlike a fixed low-pass it has no response that can be "
                        "stated in advance; this is the only honest way to "
                        "quote a retention number. Costs two extra full runs."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored result when one matches this exact "
                        "source, parameters, METHOD_VERSION and upstream "
                        "registration. Turning it off forces a fresh pass over "
                        "every frame."},
    ],
    "background": [
        {"name": "channels", "type": "str", "units": "-", "required": False,
         "default": None,
         "description": "Which channel is which, as \"dluc=2,bf=0,struct=1\". "
                        "Left out, the assignment is inferred and remembered "
                        "as a decision, so a person is asked at most once."},
        {"name": "structural_samples", "type": "int", "units": "-",
         "required": False, "default": 20,
         "description": "Frames sampled for the structural time-average. The "
                        "mask this feeds is smooth and low-frequency, so "
                        "twenty frames already average away the shot noise "
                        "that would move it."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored result when one matches this exact "
                        "source, parameters, METHOD_VERSION and upstream "
                        "registration. Turning it off forces a fresh pass over "
                        "every frame."},
    ],
    "automatic_scn_outline": [
        {"name": "scn_channel", "type": "int", "units": "-",
         "required": False, "default": None,
         "description": "One-based ImageJ channel used to calculate the "
                        "outline. For an RGB sample-axis TIFF, 1, 2 and 3 "
                        "select red, green and blue respectively. Optional for "
                        "a single-channel image; required when the source has "
                        "multiple channels."},
        {"name": "scn_z", "type": "int", "units": "-",
         "required": False, "default": None,
         "description": "One-based depth plane used to calculate the outline "
                        "time mean. Optional when there is one or no Z plane; "
                        "required for a multi-depth hyperstack."},
        {"name": "scn_time", "type": "str/int", "units": "-",
         "required": False, "default": "mean",
         "description": "Time source used for outlining: 'mean' averages the "
                        "selected channel over time, 'max' makes a per-pixel "
                        "maximum projection, and a one-based integer uses that "
                        "source frame."},
        {"name": "selected_source_only", "type": "bool", "units": "-",
         "required": False, "default": False,
         "description": "Write only the selected two-dimensional outline "
                        "plane instead of transforming every hyperstack plane. "
                        "This permits one frame and channel to be read from a "
                        "large online-only source."},
        {"name": "hash_source", "type": "bool", "units": "-",
         "required": False, "default": True,
         "description": "Calculate a SHA-256 hash over the whole source file. "
                        "Set false for a large online-only stack; the selected "
                        "outline plane and every output are still hashed."},
        {"name": "write_oriented_source", "type": "bool", "units": "-",
         "required": False, "default": True,
         "description": "Write the full oriented copy of the source stack. Set "
                        "false to keep the orientation in the report but skip a "
                        "second streaming pass over every plane and a second "
                        "full-size file, which is wasted if the next step reads "
                        "only the crop."},
        {"name": "valid_mask", "type": "path", "units": "-",
         "required": False, "default": None,
         "description": "Valid registered pixels for the time-mean image. "
                        "For meanred_<key>.tif, validfield_<key>.tif beside it "
                        "is found automatically. A missing mask is refused "
                        "unless allow_full_frame_valid is explicitly enabled."},
        {"name": "outline_roi", "type": "path", "units": "-",
         "required": False, "default": None,
         "description": "A hand-drawn outline region to use instead of "
                        "measuring the automatic outline for this recording."},
        {"name": "orient_scn", "type": "bool", "units": "-",
         "required": False, "default": True,
         "description": "By default, after drawing the accepted outline, rotate both the "
                        "two-label mask and source image so the medial gap is "
                        "vertical and inferred anatomical top is upward. "
                        "Set False to preserve the accepted source geometry."},
        {"name": "orient_up_deg", "type": "float", "units": "degrees",
         "required": False, "default": None,
         "description": "A person-supplied upward orientation angle. When "
                        "given it replaces the automatic orientation result."},
        {"name": "orient_flip", "type": "bool", "units": "-",
         "required": False, "default": False,
         "description": "Flip the manually oriented result by 180 degrees."},
        {"name": "orient_roi", "type": "path", "units": "-",
         "required": False, "default": None,
         "description": "A hand-drawn orientation region for this recording."},
        {"name": "orientation_profile_bin_px", "type": "float", "units": "px",
         "required": False, "default": 2.0,
         "description": "Width of the paired inner-edge bins used to fit the "
                        "medial gap. Higher values average more boundary detail "
                        "and can hide a small residual tilt."},
        {"name": "orientation_flare_tie_px", "type": "float", "units": "px",
         "required": False, "default": 5.0,
         "description": "Terminal gap-flare difference below which lobe-tip "
                        "alignment chooses the top. Higher values invoke the "
                        "fallback more often."},
        {"name": "crop_mode", "type": "str", "units": "-",
         "required": False, "default": "standard",
         "description": "Square crop after outlining and orientation: tight, "
                        "standard, wide, custom or none. Presets are scaled "
                        "from the smallest outline-centred square containing the "
                        "complete two-lobe mask."},
        {"name": "crop_size_px", "type": "int", "units": "px",
         "required": False, "default": None,
         "description": "Exact custom square side length centred on the "
                        "complete SCN outline. Supplying it selects custom mode. "
                        "A size that would cut the SCN outline is refused."},
        {"name": "crop_region", "type": "str", "units": "-",
         "required": False, "default": None,
         "description": "An explicit crop region supplied in the upstream "
                        "region grammar instead of an automatic preset."},
        {"name": "crop_roi", "type": "path", "units": "-",
         "required": False, "default": None,
         "description": "A hand-drawn crop region for this recording."},
        {"name": "stable_local_line_redetect", "type": "bool", "units": "-",
         "required": False, "default": True,
         "description": "Replace the original lobe line only when one-radius "
                        "and one-diameter initial-mask neighbourhoods agree on "
                        "the same changed cue and shape axes."},
        {"name": "open_enclosed_carve_channels", "type": "bool", "units": "-",
         "required": False, "default": True,
         "description": "Open only enclosed remnants of the carved medial "
                        "channel by a remove-only zero-turn pole course."},
    ],
    "segment": [
        {"name": "channels", "type": "str", "units": "-", "required": False,
         "default": None,
         "description": "Which channel is which, as \"dluc=2,bf=0,struct=1\". "
                        "Left out, the assignment is inferred and remembered "
                        "as a decision, so a person is asked at most once."},
        {"name": "structural_samples", "type": "int", "units": "-",
         "required": False, "default": 20,
         "description": "Frames sampled for the structural time-average."},
        {"name": "stationarity_check", "type": "bool", "units": "-",
         "required": False, "default": True,
         "description": "Measure how far each object's intensity centroid "
                        "travels over the recording. A cell that wanders is "
                        "not a still cell, and a static mask over a moving "
                        "object measures two different things at the two ends "
                        "of the record. Costs one pass over a small box per "
                        "object."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored result when one matches this exact "
                        "source, parameters, METHOD_VERSION and upstream "
                        "registration."},
    ],
    "export_roi": [
        {"name": "channels", "type": "str", "units": "-", "required": False,
         "default": None,
         "description": "Which channel is which, as \"dluc=2,bf=0,struct=1\"."},
        {"name": "objects", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Include one ROI per segmented object. Needs a stored "
                        "segmentation for this source."},
        {"name": "scn", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Include the SCN region and its two lobes, from the "
                        "stored decision. A hand tracing wins over the "
                        "automatic one and is never re-derived."},
        {"name": "roi_tolerance", "type": "float", "units": "px",
         "required": False, "default": 1.0,
         "description": "Polygon simplification tolerance, px. Larger drops "
                        "points from the outline; a ROI simplified past a "
                        "pixel or two stops matching the mask it came from."},
    ],
    "extract_traces": [
        {"name": "channels", "type": "str", "units": "-", "required": False,
         "default": None,
         "description": "Which channel is which, as \"dluc=0,bf=1,struct=2\"."},
        {"name": "labels", "type": "path", "units": "-", "required": False,
         "default": None,
         "description": "A label image to trace. Left out, the stored "
                        "segmentation for this source is used."},
        {"name": "baselines", "type": "list", "units": "h", "required": False,
         "default": [24.0, 48.0],
         "description": "Rolling-baseline windows to compare, in hours. A "
                        "window with no fully supported centre sample is "
                        "refused rather than filled with edge reflections."},
        {"name": "detrends", "type": "list", "units": "-", "required": False,
         "default": ["cubic", "poly6"],
         "description": "Whole-window polynomial detrends compared alongside "
                        "the rolling ones. Every method divides by the trace's "
                        "window mean, so all three land on one axis."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored result when one matches this exact "
                        "source, parameters, METHOD_VERSION and upstream "
                        "segmentation."},
    ],
    "run_controls": [
        {"name": "channels", "type": "str", "units": "-", "required": False,
         "default": None,
         "description": "Which channel is which, as \"dluc=0,bf=1,struct=2\"."},
        {"name": "labels", "type": "path", "units": "-", "required": False,
         "default": None,
         "description": "A label image to test. Left out, the stored "
                        "segmentation for this source is used."},
        {"name": "regions", "type": "dict", "units": "-", "required": False,
         "default": None,
         "description": "Named masks to measure the instrumental control "
                        "inside. Left out, the stored SCN decision is used, "
                        "and failing that the whole tissue mask."},
        {"name": "decoy_seed", "type": "int", "units": "-", "required": False,
         "default": 0,
         "description": "Decoy placement is random; the seed makes a run "
                        "repeatable. Change it to check a verdict does not "
                        "depend on where the decoys happened to land."},
        {"name": "baselines", "type": "list", "units": "h", "required": False,
         "default": [24.0, 48.0],
         "description": "Rolling-baseline windows, in hours. The first is used "
                        "for the decoy amplitude, so decoys and objects are "
                        "measured the same way."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored control when one matches this exact "
                        "source, parameters and METHOD_VERSION. On by default "
                        "because this is the slowest step in the package."},
    ],
    "test_rhythm": [
        {"name": "traces", "type": "path", "units": "-", "required": False,
         "default": None,
         "description": "Traces to test. Left out, the stored traces for this "
                        "source are used."},
        {"name": "control", "type": "mapping", "units": "-",
         "required": False, "default": None,
         "description": "The instrumental-control result that must accompany "
                        "a rhythm claim. Left out, the stored matching control "
                        "is resolved for this source."},
        {"name": "times_h", "type": "list", "units": "h", "required": False,
         "default": None,
         "description": "The time axis, when traces are passed directly."},
        {"name": "labels", "type": "list", "units": "-", "required": False,
         "default": None,
         "description": "Names for the traces, when passed directly."},
        {"name": "bin_minutes", "type": "int", "units": "min",
         "required": False, "default": 30,
         "description": "Bin the periodogram resamples to. Shorter than the "
                        "sampling interval only interpolates."},
        {"name": "fit_cosinor", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Also fit a cosinor at each trace's peak period, for "
                        "mesor, amplitude and acrophase."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use stored traces rather than re-extracting them."},
    ],
    "unmix": [
        {"name": "signal_channel", "type": "int", "units": "-",
         "required": False, "default": 2,
         "description": "One-based channel to clean."},
        {"name": "autofluorescence_channel", "type": "int", "units": "-",
         "required": False, "default": 1,
         "description": "One-based channel holding the autofluorescence that is "
                        "scaled and subtracted."},
        {"name": "reuse", "type": "bool", "units": "-", "required": False,
         "default": True,
         "description": "Use a stored result when one matches this exact "
                        "source, parameters, METHOD_VERSION and upstream "
                        "registration. Turning it off forces a fresh pass over "
                        "every frame."},
    ],
}

# ---------------------------------------------------------------- figures
# Every figure action takes these three. They are here rather than in
# COMMON_PARAMS because a measurement action has no use for a theme and no
# claim to make.
_REPROFIG_PARAMS: list[dict] = [
    {"name": "figure_profile", "type": "str", "units": "-", "required": False,
     "default": "master",
     "description": "Embedded ReproFig record profile: master, public, or "
                    "minimal_public."},
    {"name": "figure_safe_columns", "type": "list", "units": "-",
     "required": False, "default": [],
     "description": "Columns explicitly approved for a public figure and "
                    "its safe CSV."},
    {"name": "public_sources", "type": "mapping", "units": "-",
     "required": False, "default": None,
     "description": "Approved source name or hash to public URL replacements."},
    {"name": "dpi_preset", "type": "str", "units": "-", "required": False,
     "default": None,
     "description": "Named ReproFig raster resolution: screen, "
                    "continuous_tone, or line_art. Exact dpi wins."},
    {"name": "render_preset", "type": "str", "units": "-", "required": False,
     "default": None,
     "description": "Alias for dpi_preset when describing the intended "
                    "render rather than a numeric resolution."},
    {"name": "render_width_in", "type": "float", "units": "in",
     "required": False, "default": None,
     "description": "Export width in inches; supplying only width preserves "
                    "the authored aspect ratio."},
    {"name": "render_height_in", "type": "float", "units": "in",
     "required": False, "default": None,
     "description": "Export height in inches; supplying only height preserves "
                    "the authored aspect ratio."},
    {"name": "format_options", "type": "mapping", "units": "-",
     "required": False, "default": None,
     "description": "Encoder options shared by all outputs or keyed by "
                    "figure format."},
    {"name": "allow_reencode", "type": "bool", "units": "-",
     "required": False, "default": False,
     "description": "Allow ReproFig to re-encode a carrier when its metadata "
                    "cannot otherwise be embedded safely."},
    {"name": "proof", "type": "bool", "units": "-", "required": False,
     "default": False,
     "description": "Capture semantic marks and a proof root. Off preserves "
                    "the existing lightweight exact-data figure workflow."},
    {"name": "required_grades", "type": "list", "units": "-", "required": False,
     "default": [],
     "description": "Verification meanings that must pass, such as "
                    "internally_consistent or display_verified."},
    {"name": "signing_key_path", "type": "path", "units": "-", "required": False,
     "default": None,
     "description": "Protected Ed25519 signing-key file; never key contents."},
    {"name": "signing_password_env", "type": "str", "units": "-", "required": False,
     "default": None,
     "description": "Name of the environment variable holding the signing-key password."},
    {"name": "trust_policy_path", "type": "path", "units": "-", "required": False,
     "default": None,
     "description": "Offline signer trust-store file used only for explicit verification."},
    {"name": "encrypted_sections", "type": "list", "units": "-", "required": False,
     "default": [],
     "description": "Evidence section identities to encrypt before the artifact is signed."},
    {"name": "encryption_password_env", "type": "str", "units": "-", "required": False,
     "default": None,
     "description": "Name of the environment variable holding a section password."},
    {"name": "recipient_file", "type": "path", "units": "-", "required": False,
     "default": None,
     "description": "JSON mapping of approved recipient names to public encryption keys."},
    {"name": "broker_policy_path", "type": "path", "units": "-", "required": False,
     "default": None,
     "description": "Controlled-output broker policy used for explicit promotion."},
]

_FIGURE_COMMON: list[dict] = [
    {"name": "theme", "type": "str", "units": "-", "required": False,
     "default": "pyflash",
     "description": "The house look, by name, from analysis-kit. 'engine' is "
                    "the deliberate exception: it leaves Matplotlib's own "
                    "defaults alone so a ported figure comes out pixel-for-"
                    "pixel like the script it replaces."},
    {"name": "dpi", "type": "int", "units": "-", "required": False,
     "default": 150,
     "description": "Raster resolution. 150 for review, 300+ for print; file "
                    "size grows with the square."},
    {"name": "output_formats", "type": "list", "units": "-",
     "required": False, "default": ["png"],
     "description": "Direct ReproFig copies. Supports SVG, PDF, PNG, JPEG, "
                    "TIFF, WebP, AVIF and HEIF; every copy keeps one figure "
                    "identity."},
    *_REPROFIG_PARAMS,
    {"name": "claim", "type": "str", "units": "-", "required": False,
     "default": "",
     "description": "The one sentence this figure proves, written into the "
                    "provenance bundle. Left empty, one is generated from what "
                    "was actually drawn."},
]

_FIGURE_SIZE: list[dict] = [
    {"name": "fig_width_in", "type": "float", "units": "-", "required": False,
     "default": 15.0, "description": "Figure width, inches."},
    {"name": "fig_height_in", "type": "float", "units": "-", "required": False,
     "default": 5.4, "description": "Figure height, inches."},
]

_CHANNELS = {
    "name": "channels", "type": "str", "units": "-", "required": False,
    "default": None,
    "description": 'Which channel is which, as "dluc=2,bf=0,struct=1". Left '
                   "out, the assignment is inferred and remembered as a "
                   "decision, so a person is asked at most once."}

_BACKGROUND = {
    "name": "background", "type": "array", "units": "counts", "required": False,
    "default": None,
    "description": "The image the outlines are drawn over - normally the "
                   "accumulated profile the segmenter thresholded, so the "
                   "picture shows the image the decision was made on. Left "
                   "out, frame 0 is used and the panel and the provenance both "
                   "say so."}

_STRUCTURAL = {
    "name": "structural", "type": "array", "units": "counts", "required": False,
    "default": None,
    "description": "The structural-channel image to add as a first panel. "
                   "Without it the figure cannot answer whether the objects "
                   "are on tissue, so the panel is left out rather than faked."}

EXTRA_PARAMS["trace_panel"] = [
    {"name": "colour_cycle", "type": "list", "units": "-", "required": False,
     "default": [],
     "description": "Colours for the 2nd and later traces merged onto one "
                    "panel, by house name or by value. Empty uses the house "
                    "reporter cycle - dLuc purple, RFP red, GFP green, then "
                    "three fillers - which is what the engine spelled out in "
                    "hex. Reordering it recolours every merged panel that "
                    "names no colour of its own."},
    {"name": "write_bundle", "type": "bool", "units": "-", "required": False,
     "default": True,
     "description": "Also write the figure as a plot-that provenance bundle: "
                    "the vector figure, the exact plotted table, copies of "
                    "every source CSV with its SHA256, and a README."},
    {"name": "claim", "type": "str", "units": "-", "required": False,
     "default": "",
     "description": "The one sentence this figure proves, written into the "
                    "provenance bundle. Left empty, one is generated from what "
                    "was actually drawn."},
    {"name": "theme", "type": "str", "units": "-", "required": False,
     "default": "engine",
     "description": "The house look, by name, from analysis-kit. 'engine' is "
                    "the deliberate exception: it leaves Matplotlib's own "
                    "defaults alone so a ported figure comes out pixel-for-"
                    "pixel like the script it replaces."},
    *_REPROFIG_PARAMS,
]

_PUBLICATION_WORKBOOK_PARAMS: list[dict] = [
    {"name": "source", "type": "path", "units": "-", "required": False,
     "default": None, "description": "One figure artifact or folder to include."},
    {"name": "artifacts", "type": "list", "units": "-", "required": False,
     "default": [], "description": "Additional figure artifacts or folders to combine."},
    {"name": "output_path", "type": "path", "units": "-", "required": True,
     "default": None, "description": "Destination canonical Excel workbook."},
    {"name": "statistics_ledger_path", "type": "path", "units": "-", "required": False,
     "default": None, "description": "Optional complete experiment statistics JSON or CSV."},
    {"name": "profile", "type": "str", "units": "-", "required": False,
     "default": "master", "description": "Master, public or minimal_public workbook profile."},
    {"name": "safe_columns", "type": "mapping", "units": "-", "required": False,
     "default": None, "description": "Per-table public column allowlists."},
    {"name": "public_sources", "type": "mapping", "units": "-", "required": False,
     "default": None, "description": "Approved source identifiers to public URLs."},
    {"name": "declare_ledger_complete", "type": "bool", "units": "-", "required": False,
     "default": False, "description": "Declare that the supplied ledger lists every analysis test."},
    {"name": "overwrite", "type": "bool", "units": "-", "required": False,
     "default": False, "description": "Replace an existing workbook atomically."},
]

EXTRA_PARAMS["registration_figure"] = _FIGURE_COMMON + [
    {"name": "shifts", "type": "table", "units": "-", "required": False,
     "default": None,
     "description": "The per-frame shift and residual table to draw. Left "
                    "out, the stored registration artefact for this source is "
                    "used, which is the usual call."},
    {"name": "fig_width_in", "type": "float", "units": "-", "required": False,
     "default": 11.0, "description": "Figure width, inches."},
    {"name": "panel_height_in", "type": "float", "units": "-",
     "required": False, "default": 2.2,
     "description": "Height of one panel, inches."},
]

EXTRA_PARAMS["cosmic_ray_preview"] = _FIGURE_COMMON + _FIGURE_SIZE + [
    {"name": "frame", "type": "int", "units": "frames", "required": False,
     "default": None,
     "description": "Which frame to show. Left out, the one the stored event "
                    "table says lost the most pixels - a preview of a quiet "
                    "frame proves nothing."},
    {"name": "channel", "type": "int", "units": "-", "required": False,
     "default": 0,
     "description": "Which channel to show, zero-based. The filter runs on "
                    "the bioluminescence channel."},
    {"name": "cleaned", "type": "path", "units": "-", "required": False,
     "default": None,
     "description": "The cleaned stack, as a path or an array, for the "
                    "after panel. Without it only the before panel and the "
                    "replacement mask are drawn."},
    {"name": "mask", "type": "array", "units": "-", "required": False,
     "default": None,
     "description": "The replacement mask. Left out, the stored one for this "
                    "source is used."},
]

EXTRA_PARAMS["channel_figure"] = _FIGURE_COMMON + _FIGURE_SIZE + [
    dict(_CHANNELS),
    {"name": "frame", "type": "int", "units": "frames", "required": False,
     "default": 0, "description": "Which frame to show, zero-based."},
]

EXTRA_PARAMS["frames_figure"] = _FIGURE_COMMON + [
    dict(_CHANNELS),
    {"name": "fig_width_in", "type": "float", "units": "-", "required": False,
     "default": 16.0, "description": "Figure width, inches."},
    {"name": "fig_height_in", "type": "float", "units": "-", "required": False,
     "default": 9.0, "description": "Figure height, inches."},
]

EXTRA_PARAMS["cell_overlay"] = _FIGURE_COMMON + [
    {"name": "labels", "type": "array", "units": "-", "required": False,
     "default": None,
     "description": "The label image to outline. Left out, the stored "
                    "segmentation for this source is used."},
    dict(_BACKGROUND),
    dict(_STRUCTURAL),
    {"name": "rings", "type": "array", "units": "-", "required": False,
     "default": None,
     "description": "The local background ring each trace was measured "
                    "against, as a label image. Without it the third panel is "
                    "dropped: an absent ring and a ring of zero area are "
                    "different facts."},
    {"name": "candidates", "type": "list", "units": "-", "required": False,
     "default": [],
     "description": "Labels that are permissive sweep candidates rather than "
                    "prominence-segmented cells. Drawn dashed."},
    {"name": "fig_width_in", "type": "float", "units": "-", "required": False,
     "default": 19.0, "description": "Figure width, inches."},
    {"name": "fig_height_in", "type": "float", "units": "-", "required": False,
     "default": 6.5, "description": "Figure height, inches."},
]

EXTRA_PARAMS["roi_overlay"] = _FIGURE_COMMON + [
    {"name": "polygons", "type": "list", "units": "-", "required": False,
     "default": [],
     "description": "The regions to draw, as polygons from pymicroglia.roi."},
    dict(_BACKGROUND),
    dict(_STRUCTURAL),
    {"name": "fig_width_in", "type": "float", "units": "-", "required": False,
     "default": 13.0, "description": "Figure width, inches."},
    {"name": "fig_height_in", "type": "float", "units": "-", "required": False,
     "default": 6.5, "description": "Figure height, inches."},
]

#: What a pipeline run needs on top of its engine's settings block: the run
#: folder's policy, the review it may be handed, and the command-line flags
#: ``dluc_pipeline.py`` declared in ``argparse`` rather than in its parameter
#: block. Shared by every pipeline, because "what happens if this run folder
#: already exists" has to mean the same thing in all of them.
_PIPELINE_RUN = [
    {"name": "if_exists", "type": "str", "units": "-", "required": False,
     "default": "version",
     "description": "What to do about a run folder that already exists: "
                    "'version' keeps both by adding _v2, 'overwrite' replaces "
                    "it, 'error' refuses, 'skip' returns the previous manifest "
                    "without recomputing. Defaults to 'version' because "
                    "overwriting is the policy that costs somebody the "
                    "comparison they were in the middle of making."},
    {"name": "run_label", "type": "str", "units": "-", "required": False,
     "default": None,
     "description": "Name for this run's folder. Left out, it is a short hash "
                    "of the settings, so an identical re-run lands in the same "
                    "place and a changed one lands beside it."},
    {"name": "review", "type": "object", "units": "-", "required": False,
     "default": None,
     "description": "A pymicroglia.review.Review to record judgement calls "
                    "into, so several runs can share one. Left out, the run "
                    "makes its own and loads any answers already stored "
                    "against this source."},
    {"name": "claim", "type": "str", "units": "-", "required": False,
     "default": "",
     "description": "The one sentence this run is meant to support, written "
                    "into the run record."},
    {"name": "reuse", "type": "bool", "units": "-", "required": False,
     "default": True,
     "description": "Read stored artefacts instead of recomputing them. Off "
                    "forces every expensive step to run again, which is what "
                    "to do after a genuine cache concern and not before one."},
]

EXTRA_PARAMS["dluc_single_cell"] = _PIPELINE_RUN + [
    {"name": "t0", "type": "float", "units": "hours", "required": False,
     "default": 72.0,
     "description": "Window start. The default exists because in the reference "
                    "dataset the weakest cell's ring-subtracted signal is "
                    "negative before 72 h; on a shorter recording it is simply "
                    "lost data, and the review says how many frames it cost."},
    {"name": "t1", "type": "float", "units": "hours", "required": False,
     "default": None,
     "description": "Window end. Left out, the usable block runs to its end. "
                    "Given, it selects the continuous block containing that "
                    "window rather than the longest one, so matched recordings "
                    "stay on matched hours."},
    {"name": "channels", "type": "str", "units": "-", "required": False,
     "default": None,
     "description": "Channel roles as 'dluc=0,bf=1,struct=2', overriding the "
                    "pixel statistics. Recorded as a decision, so it is not "
                    "asked again."},
    {"name": "baselines", "type": "list", "units": "hours", "required": False,
     "default": [24.0, 48.0],
     "description": "Rolling-baseline lengths. Each one truncates half its "
                    "own length at each end of the window, which the review "
                    "flags when it eats most of the record."},
    {"name": "detrends", "type": "list", "units": "-", "required": False,
     "default": ["cubic", "poly6"],
     "description": "Whole-window polynomial cross-checks. 'bicubic' is an "
                    "accepted alias for 'cubic'; the trace is one-dimensional, "
                    "so the operation is a cubic in time and not a "
                    "two-dimensional image interpolation."},
    {"name": "roi", "type": "path", "units": "-", "required": False,
     "default": None,
     "description": "A hand-drawn .roi or .zip to check the automatic outline "
                    "against. Below roi_dice_min the hand tracing wins and the "
                    "disagreement is recorded as a blocker."},
    {"name": "shift_mode", "type": "str", "units": "-", "required": False,
     "default": "integer",
     "description": "'integer' rolls whole pixels and leaves every value "
                    "exactly as measured, which is what any spatial statistic "
                    "needs. 'subpixel' interpolates and is marginally better "
                    "for intensity alone."},
    {"name": "seed", "type": "int", "units": "-", "required": False,
     "default": 163,
     "description": "Decoy placement is random; this is what makes a p-value "
                    "reproducible."},
    {"name": "dt_min", "type": "float", "units": "minutes", "required": False,
     "default": None,
     "description": "Fallback frame spacing when the file carries no plane "
                    "timestamps. A uniform assumption cannot find a pause in "
                    "the recording, only hide it."},
    {"name": "um_per_px", "type": "float", "units": "um", "required": False,
     "default": None,
     "description": "Pixel size, when the file is not calibrated. Areas stay "
                    "in pixels without it."},
    {"name": "skip_control", "type": "bool", "units": "-", "required": False,
     "default": False,
     "description": "Skip the instrumental control. Never do this before a "
                    "periodicity or circadian claim: skipping it is recorded "
                    "as a blocker, because nothing else in the run "
                    "distinguishes a rhythm in the sample from one in the "
                    "microscope."},
    {"name": "skip_videos", "type": "bool", "units": "-", "required": False,
     "default": False,
     "description": "Skip the time-resolved visual quality-control videos. "
                    "Mask acceptance is then incomplete, because whether a "
                    "static outline stays on the same feature is a question "
                    "only a movie answers."},
    {"name": "video_fps", "type": "float", "units": "fps", "required": False,
     "default": None,
     "description": "Explicit quality-control frame rate. Normally left out "
                    "and derived from video_hours_per_second and the "
                    "acquisition rate, so playback means the same biological "
                    "speed on every recording."},
    {"name": "publication_fps", "type": "float", "units": "fps",
     "required": False, "default": None,
     "description": "Explicit publication frame rate. Normally left out, for "
                    "the same reason as video_fps."},
    {"name": "work", "type": "path", "units": "-", "required": False,
     "default": None,
     "description": "Accepted for compatibility with the engine's --work "
                    "flag and not used: the registered arrays are keyed "
                    "artefacts in the project's own PixelStore, which is "
                    "already capped and already evicts."},
    {"name": "cosmic_seed_z", "type": "float", "units": "noise units",
     "required": False, "default": 12.0,
     "description": "How far above the mean of the two neighbouring frames a "
                    "pixel must sit to be called an outlier, in this "
                    "recording's own noise. CAUTION: this is the setting that "
                    "decides what counts as data. Replaced cosmic_k on "
                    "2026-08-20 with the method itself; the number is the same "
                    "and what it is measured against is not."},
    {"name": "cosmic_growth_px", "type": "int", "units": "px",
     "required": False, "default": 2,
     "description": "How far each outlier is dilated before repair, so the "
                    "faint skirt around a hit goes with it. Replaced "
                    "cosmic_grow on 2026-08-20."},
]

EXTRA_PARAMS["cry1_dluc_photon"] = _PIPELINE_RUN + [
    {"name": "videos", "type": "bool", "units": "-", "required": False,
     "default": True,
     "description": "Render the four photon-aware movies. Off stops after the "
                    "registered TIFF, which is the only output a number should "
                    "come from anyway."},
]

#: Parameter names an action's source block declares that the action does not
#: take. Keyed by action name. Every entry is a setting belonging to a later
#: stage or to a Fiji wrapper, and dropping it here is what keeps ``describe``
#: honest: a parameter an agent is shown is one the action actually accepts.
DROPPED_PARAMS: dict[str, set[str]] = {
    # ``microglia_red_only_video_export.py`` is one script doing two jobs. The
    # unmixing is a keyed measurement step; everything else in its block is the
    # video it then renders, which is stage 10's.
    "unmix": {"fps", "frame_interval_minutes", "time_label_band_height",
              "display_percentile", "crf", "file_lock_retry_seconds",
              "timestamp_font_candidates", "timestamp_font_fallback",
              "timestamp_font_height_fraction", "timestamp_font_min_size"},
    # PyMicroglia vendors the accepted engine. These three parameters belong to
    # the standalone protocol's dynamic-import guard, not to the self-contained
    # action; the source and port hashes remain in every generated report.
    "automatic_scn_outline": {
        "implementation_dir", "expected_scn_roi_sha256",
        "expected_red_measure_sha256",
    },
}

#: ``dluc_pipeline.py`` is one 3600-line script holding the whole single-cell
#: analysis, so its parameter block covers everything from the cosmic filter to
#: the video codec. Three actions are cut out of it, and each keeps only the
#: names it actually uses — otherwise ``describe segment`` would offer an agent
#: a video's frame rate.
_DLUC = "Analysis/dLuc_single_cell_analysis/dluc_pipeline.py"
_DLUC_ALL = {
    "gap_h", "crop_pad", "cosmic_k", "cosmic_grow", "tissue_pct",
    "tissue_smooth", "off_dilate", "prof_smooth", "k_mask", "k_soma",
    "prominence", "k_relax", "relax_below", "k_detect", "k_cand", "minsep",
    "min_cand_px", "ring_in", "ring_out", "ring_wide", "ring_min", "move_max",
    "cent_box", "ndecoy", "decoy_p", "smooth_display", "poly_edge_h",
    "video_hours_per_second", "publication_band", "dluc_video_temporal",
    "dluc_video_spatial", "dluc_video_noise_pct", "ls_pmin", "ls_pmax", "ls_n",
    "ls_rhythmic", "roi_smooth", "roi_area_lo", "roi_area_hi", "roi_dice_min",
    "font_regular_path", "font_bold_path",
}
_BACKGROUND_KEEPS = {"tissue_pct", "tissue_smooth", "off_dilate", "prof_smooth"}
_SEGMENT_KEEPS = _BACKGROUND_KEEPS | {
    "k_mask", "k_soma", "prominence", "k_relax", "relax_below", "k_detect",
    "k_cand", "minsep", "min_cand_px", "move_max", "cent_box"}
#: export_roi thresholds nothing — it writes a zip from a stored
#: segmentation and a stored decision. The ROI-automation settings belong
#: to roi.scn_roi, which is called during segmentation, not here.
_ROI_KEEPS: set[str] = set()
_TRACES_KEEPS = {"ring_in", "ring_out", "ring_wide", "ring_min",
                 "smooth_display", "poly_edge_h"}
_CONTROLS_KEEPS = {"ndecoy", "decoy_p", "ls_pmin", "ls_pmax", "ls_rhythmic"}
_RHYTHM_KEEPS = {"ls_pmin", "ls_pmax", "ls_n", "ls_rhythmic"}

#: ``run_trace_panel_figure.ps1`` declares the packages it hunts for before
#: handing off to Python. That is the wrapper's environment check, not a
#: setting of the figure, and an agent offered it would have nothing to do
#: with it.
DROPPED_PARAMS["trace_panel"] = {"requiredpackages"}

#: A default this package deliberately spells differently from the engine.
#: Applied after the harvest, so the parameter keeps its name, its type and its
#: prose and changes only the value it starts at.
#:
#: All three figure colours become *names*: ``colour("dluc")`` is ``#a340d1``
#: and ``GREY_LEVEL["raw"]`` is ``"0.72"``, so the figure is unchanged and the
#: value is written down in one place instead of two. A hex literal in a
#: default is how the house style came to have four different reds.
DEFAULT_OVERRIDES: dict[str, dict[str, object]] = {
    "trace_panel": {
        "colour": "dluc",
        "raw_colour": "raw",
        "shade_colour": "shade",
    },
}

#: Actions whose whole harvested block belongs to another stage. The source is
#: recorded for provenance — these figures reproduce what those engines drew —
#: but none of the engine's own settings is a setting of the figure.
DROP_EVERYTHING = "*"

#: ``dluc_pipeline.py`` still declares the cosmic filter it was written
#: against. This package's dLuc pipeline moved to the one-outlier rule on
#: 2026-08-20, and its two settings are declared under their new names in
#: ``EXTRA_PARAMS``. Dropping the old names is what keeps ``describe`` honest:
#: a parameter an agent is shown is one the action actually accepts.
DROPPED_PARAMS["dluc_single_cell"] = {"cosmic_k", "cosmic_grow"}

DROPPED_PARAMS["background"] = _DLUC_ALL - _BACKGROUND_KEEPS
DROPPED_PARAMS["segment"] = _DLUC_ALL - _SEGMENT_KEEPS
DROPPED_PARAMS["export_roi"] = _DLUC_ALL - _ROI_KEEPS
DROPPED_PARAMS["extract_traces"] = _DLUC_ALL - _TRACES_KEEPS
DROPPED_PARAMS["run_controls"] = _DLUC_ALL - _CONTROLS_KEEPS
DROPPED_PARAMS["test_rhythm"] = _DLUC_ALL - _RHYTHM_KEEPS

# The four quality-control figures and the two overlays reproduce what those
# engines drew, so the engine is recorded as their source. None of the engine's
# own settings is a setting of the figure: a preview of a cosmic-ray mask has no
# threshold, because the thresholding already happened and is what it is drawing.
for _figure in ("registration_figure", "cosmic_ray_preview", "channel_figure",
                "frames_figure", "cell_overlay", "roi_overlay"):
    DROPPED_PARAMS[_figure] = DROP_EVERYTHING

#: The actions PyMicroglia exposes, and where each was copied from.
#:
#: ``method`` is the dotted target inside this package. Most do not exist yet;
#: an action whose target is missing is reported by ``discover`` as pending,
#: which is how a half-ported package stays a visible state rather than a
#: surprise. ``source`` is provenance only — nothing reads it at run time.
ACTIONS: list[dict] = [
    {
        "name": "register",
        "method": "registration.estimate_and_apply",
        "mutates": True,
        "summary": "Register a time-lapse by translation-only phase correlation on a "
                   "stable channel, applied to every channel at full resolution.",
        "source": ["Analysis/microglia_phase_correlation_registration.py",
                   "Analysis/microglia_phase_correlation_registration_analysis.ijm"],
    },
    {
        "name": "register_three_channel",
        "method": "registration.estimate_and_apply_three_channel",
        "mutates": True,
        "summary": "Register three-channel phase/green/red organotypic time-lapses on "
                   "the red neuronal channel.",
        "source": ["Analysis/phase_green_red_timelapse_pipeline.py",
                   "Analysis/phase_green_red_registration_analysis.ijm"],
    },
    {
        "name": "export_registered_stack",
        "method": "registration.export_registered_stack",
        "mutates": True,
        "summary": "Export cropped registered raw two-channel stacks with no unmixing, "
                   "normalisation or rescaling.",
        "source": ["Analysis/microglia_raw_registered_stack_export.py",
                   "Analysis/microglia_raw_registered_stack_analysis.ijm"],
    },
    {
        "name": "remove_cosmic_rays",
        "method": "cosmic.remove_cosmic_rays",
        "mutates": True,
        "summary": "Replace cosmic-ray damage using one outlier rule asked of a "
                   "pixel, of a line and of the bleed off a saturated pixel, "
                   "after registration.",
        "source": ["Analysis/microglia_cosmic_ray_removal.py",
                   "Analysis/microglia_cosmic_ray_removal_analysis.ijm"],
    },
    {
        "name": "remove_static_background",
        "method": "display.remove_static_background",
        "mutates": True,
        "display_only": True,
        "summary": "Remove the static background and shot noise for display only, "
                   "preserving pulse amplitude at circadian periods.",
        "source": ["Analysis/microglia_static_background_removal.py",
                   "Analysis/microglia_static_background_removal_analysis.ijm"],
    },
    {
        "name": "display_filter",
        "method": "display.bioluminescence_display",
        "mutates": True,
        "display_only": True,
        "summary": "Minimally filtered bioluminescence display that keeps the background "
                   "visible so the channel merges without looking clipped; display only.",
        "source": ["Analysis/microglia_bioluminescence_display.py",
                   "Analysis/microglia_bioluminescence_display_analysis.ijm"],
    },
    {
        "name": "background",
        "method": "segmentation.background",
        "mutates": True,
        "summary": "Estimate the off-tissue background from the structural channel, "
                   "never from the image corners, and check the tissue mask actually "
                   "contains the bioluminescence.",
        "source": [_DLUC],
    },
    {
        "name": "segment",
        "method": "segmentation.segment",
        "mutates": True,
        "summary": "Detect still cells by soma prominence and watershed, then sweep for "
                   "further candidates. No minimum cell area, by rule.",
        "source": [_DLUC],
    },
    {
        "name": "automatic_scn_outline",
        "method": "scn_outline.automatic_scn_outline",
        "mutates": True,
        "summary": "Draw the accepted A007 two-lobe SCN outline from a "
                   "registered red-channel time mean or selected hyperstack "
                   "channel using a mean, maximum projection or chosen frame "
                   "and valid-field mask, either retain that selected plane or "
                   "stream one transform across every stack plane, "
                   "apply the accepted A006 top-up orientation by default, "
                   "and write a mask-relative square crop.",
        "source": ["Analysis/automatic_scn_roi/automatic_scn_roi.py"],
    },
    {
        "name": "export_roi",
        "method": "roi.export_roi",
        "mutates": True,
        "summary": "Write a RoiSet.zip Fiji can open: one region per segmented object, "
                   "plus the SCN region and its two lobes.",
        "source": [_DLUC],
    },
    {
        "name": "extract_traces",
        "method": "tracing.extract_traces",
        "mutates": True,
        "summary": "One trace per segmented object, with its local ring subtracted, "
                   "and dF/F against each trace's window mean rather than an "
                   "instantaneous rolling baseline.",
        "source": [_DLUC],
    },
    {
        "name": "run_controls",
        "method": "controls.run_controls",
        "mutates": True,
        "summary": "Area-matched decoys placed on tissue and read in absolute counts, "
                   "plus the instrumental control that tests whether a rhythm is also "
                   "off tissue, in another channel, or in image sharpness.",
        "source": [_DLUC],
    },
    {
        "name": "test_rhythm",
        "method": "rhythm.test_rhythm",
        "mutates": False,
        "summary": "Period and cosinor for every trace, from circadian-workbench, and "
                   "refused outright for a source with no instrumental control.",
        "source": [_DLUC],
    },
    {
        "name": "unmix",
        "method": "filtering.unmix",
        "mutates": True,
        "summary": "Linearly unmix a signal channel from a scaled autofluorescence "
                   "channel, per frame, with one constant coefficient for the whole "
                   "recording.",
        # Not a protocol of its own. The formula lives inside the red-only video
        # export, and pulling it out is deliberate: that script unmixes and then
        # renders, while microglia_raw_registered_stack_export.py exports the
        # same recording and does not unmix at all. That difference has to stay
        # visible, which means unmixing is a step a caller opts into by name
        # rather than something a series does to itself on load.
        "source": ["Analysis/microglia_red_only_video_export.py"],
    },
    {
        "name": "red_only_video",
        "method": "video.red_only",
        "mutates": True,
        "display_only": True,
        "summary": "Linearly unmix microglial mCherry from autofluorescence on every raw "
                   "frame, apply the saved registration transform, and export registered "
                   "two-channel stacks plus timestamped red-only videos.",
        "source": ["Analysis/microglia_red_only_video_export.py"],
    },
    {
        "name": "composite_video",
        "method": "video.timestamped_composite",
        "mutates": True,
        "display_only": True,
        "summary": "Render timestamped red/green microglia composite videos from registered "
                   "two-channel stacks, correcting autofluorescence drift with a "
                   "display-only endpoint gain curve.",
        "source": ["Analysis/microglia_timestamped_composite_video_export.py"],
    },
    {
        "name": "phase_green_red_video",
        "method": "video.phase_green_red",
        "mutates": True,
        "display_only": True,
        "summary": "Render timestamped green/red videos from a registered three-channel "
                   "organotypic time-lapse and audit which channel is which, leaving the "
                   "scientific data unnormalised.",
        "source": ["Analysis/phase_green_red_video_export.py"],
    },
    {
        "name": "stack_to_mp4",
        "method": "video.stack_to_mp4",
        "mutates": True,
        "display_only": True,
        "summary": "Render a TIFF time-lapse stack as an MP4 at a stated number of "
                   "experimental hours per second, through a named colour map.",
        "source": ["Analysis/tiff_stack_to_mp4.py", "Analysis/tiff_stack_to_mp4.ijm"],
    },
    {
        "name": "trace_panel",
        "method": "trace_tables.trace_panel",
        "mutates": True,
        "summary": "Draw a stack of time-trace panels from one or more trace CSVs, choosing "
                   "which trace goes in which panel, merging several onto one panel, with "
                   "independent control of colours, detrend, grey underlay, edge shading, "
                   "time window, x ticks and vertical lines.",
        "source": ["Analysis/trace_panel_figure/trace_panel_figure.py",
                   "Analysis/trace_panel_figure/run_trace_panel_figure.ps1"],
    },
    {
        "name": "dluc_single_cell",
        "method": "pipelines.dluc_single_cell.run",
        "mutates": True,
        "summary": "Segment and trace single-cell dLuc bioluminescence from a registered "
                   "time-lapse, with detrending, rhythmicity tests, instrumental controls, "
                   "quality-control figures and publication videos.",
        "source": ["Analysis/dLuc_single_cell_analysis/dluc_pipeline.py",
                   "Analysis/dLuc_single_cell_analysis/run_dluc_analysis.ps1",
                   "Analysis/dLuc_single_cell_analysis/run_dluc_batch.ps1",
                   "Analysis/dLuc_single_cell_analysis/setup_environment.ps1"],
    },
    {
        "name": "cry1_dluc_photon",
        "method": "pipelines.cry1_dluc_photon.run",
        "mutates": True,
        "summary": "Register four-channel Cry1-dLuc time-lapses on the bright-field channel "
                   "and render photon-aware bioluminescence videos and quality-control "
                   "figures, leaving the saved TIFF as raw transformed counts.",
        "source": ["Analysis/cry1_dluc_photon_pipeline.py"],
    },
    {
        "name": "registration_figure",
        "method": "visualisation.qc.registration_figure",
        "mutates": True,
        "summary": "Draw the per-frame translation registration applied and the "
                   "residual it left, from the stored shift table rather than from "
                   "pixels.",
        "source": ["Analysis/microglia_phase_correlation_registration.py"],
    },
    {
        "name": "cosmic_ray_preview",
        "method": "visualisation.qc.cosmic_ray_preview",
        "mutates": True,
        "summary": "Draw one frame before and after cosmic-ray removal beside the "
                   "replacement mask, choosing the frame that lost the most pixels.",
        "source": ["Analysis/microglia_cosmic_ray_removal.py"],
    },
    {
        "name": "channel_figure",
        "method": "visualisation.qc.channel_figure",
        "mutates": True,
        "summary": "Draw each channel's frame and intensity histogram side by side, so "
                   "a swapped or bleeding channel is visible before anything is measured.",
        "source": ["Analysis/cry1_dluc_photon_pipeline.py"],
    },
    {
        "name": "frames_figure",
        "method": "visualisation.qc.frames_figure",
        "mutates": True,
        "summary": "Draw the first, middle and last frame of every channel with its "
                   "timestamp, which catches a wrongly ordered hyperstack in three "
                   "plane reads.",
        "source": ["Analysis/cry1_dluc_photon_pipeline.py"],
    },
    {
        "name": "cell_overlay",
        "method": "visualisation.overlays.cell_overlay",
        "mutates": True,
        "summary": "Draw the segmented objects as outlines over the tissue and the "
                   "bioluminescence they were found in, with each object's local "
                   "background ring.",
        "source": ["Analysis/dLuc_single_cell_analysis/dluc_pipeline.py"],
    },
    {
        "name": "roi_overlay",
        "method": "visualisation.overlays.roi_overlay",
        "mutates": True,
        "summary": "Draw stored regions over the frame they were drawn on, so a "
                   "region decision can be checked once instead of re-made every run.",
        "source": ["Analysis/dLuc_single_cell_analysis/dluc_pipeline.py"],
    },
]


def build(protocols: Path) -> dict:
    """Harvest every action's sources into one catalogue.

    One parameter name means one thing across the project — the kit's registry
    enforces that — but the same name genuinely carries different defaults in
    different protocols (``compression_level``, ``crf``, ``fps``). So the shared
    vocabulary holds the canonical type, units and prose, and each action keeps
    its own defaults. ``describe`` merges the two, and an agent sees the default
    that actually applies to the action it asked about.
    """
    vocabulary: dict[str, dict] = {}
    conflicts: list[str] = []
    actions: list[dict] = []

    common_defaults = {row["name"]: row["default"] for row in COMMON_PARAMS}
    for row in COMMON_PARAMS:
        vocabulary[row["name"]] = {k: v for k, v in row.items() if k != "default"}

    for entry in ACTIONS:
        paths = [protocols / rel for rel in entry["source"]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(
                f"{entry['name']}: source not found: {[str(p) for p in missing]}\n"
                "Pass --protocols to point at the folder these were copied from."
            )
        block = params.harvest_many(paths)

        dropped = DROPPED_PARAMS.get(entry["name"], set())
        overrides = DEFAULT_OVERRIDES.get(entry["name"], {})
        defaults: dict[str, object] = dict(common_defaults)
        names: list[str] = [row["name"] for row in COMMON_PARAMS]
        for doc in block.params:
            if doc.name in common_defaults:
                continue          # the common key wins; blocks vary in wording
            if dropped == DROP_EVERYTHING or doc.name in dropped:
                continue          # belongs to a later stage or a Fiji wrapper
            names.append(doc.name)
            defaults[doc.name] = overrides.get(doc.name, doc.default)
            shared = {"name": doc.name, "type": doc.type, "units": doc.units,
                      "description": doc.description, "required": doc.required}
            known = vocabulary.get(doc.name)
            if known is None:
                vocabulary[doc.name] = shared
            elif known != shared:
                conflicts.append(f"{entry['name']}.{doc.name}")

        for extra in EXTRA_PARAMS.get(entry["name"], []):
            if extra["name"] in defaults:
                continue
            names.append(extra["name"])
            defaults[extra["name"]] = extra["default"]
            shared = {k: v for k, v in extra.items() if k != "default"}
            known = vocabulary.get(extra["name"])
            if known is None:
                vocabulary[extra["name"]] = shared
            elif known != shared:
                conflicts.append(f"{entry['name']}.{extra['name']}")

        actions.append({
            "name": entry["name"],
            "summary": entry["summary"],
            "method": entry["method"],
            "mutates": entry.get("mutates", False),
            "destructive": entry.get("destructive", False),
            "display_only": entry.get("display_only", False),
            "params": names,
            "defaults": defaults,
            "method_version": METHOD_VERSION_OVERRIDES.get(
                entry["name"], block.method_version),
            "source": entry["source"],
        })

    workbook_defaults = {
        row["name"]: row["default"] for row in _PUBLICATION_WORKBOOK_PARAMS
    }
    for row in _PUBLICATION_WORKBOOK_PARAMS:
        shared = {key: value for key, value in row.items() if key != "default"}
        known = vocabulary.get(row["name"])
        if known is None:
            vocabulary[row["name"]] = shared
        elif known != shared:
            conflicts.append(f"publication_workbook.{row['name']}")
    actions.append({
        "name": "publication_workbook",
        "summary": "Combine ReproFig figure data and every declared statistical test into one verified journal Excel workbook.",
        "method": "publication.publication_workbook",
        "mutates": True,
        "destructive": False,
        "display_only": False,
        "params": [row["name"] for row in _PUBLICATION_WORKBOOK_PARAMS],
        "defaults": workbook_defaults,
        "method_version": "1",
        "source": [],
    })

    return {
        "project": "pymicroglia",
        "generated_by": "tools/regenerate_catalogue.py",
        "actions": actions,
        "params": [vocabulary[name] for name in sorted(vocabulary)],
        # Names whose prose differs between protocols. The first occurrence's
        # wording is the one kept; per-action defaults are never lost.
        "vocabulary_conflicts": sorted(set(conflicts)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocols", type=Path, default=DEFAULT_PROTOCOLS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the shipped catalogue is out of date")
    args = parser.parse_args(argv)

    catalogue = build(args.protocols)
    text = json.dumps(catalogue, indent=2, sort_keys=False, default=str) + "\n"

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != text:
            print("catalogue is out of date; run tools/regenerate_catalogue.py")
            return 1
        print("catalogue is up to date")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"  {len(catalogue['actions'])} actions, "
          f"{len(catalogue['params'])} distinct parameters, "
          f"{len(catalogue['vocabulary_conflicts'])} wording conflicts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
