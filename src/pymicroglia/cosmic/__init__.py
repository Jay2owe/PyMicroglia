"""Cosmic-ray damage, removed by one rule asked three times.

Three parts, and they answer different questions.

``rule``   what an outlier *is* — the reference, the noise, the z, the line
           test and the bleed model. Pure arithmetic on arrays, with no file in
           sight, which is what lets the engine's own test cases be copied
           across rather than approximated.
``clean``  what to *do* about one: walk the stack twice, repair what the rule
           found, and write the cleaned copy with its mask, its event table and
           its bleed profile.
``stack``  the same, for a series already in memory and no file to write —
           what the single-cell dLuc pipeline has when it reaches this step.

The two entry points share one ``METHOD_VERSION`` because they are one method:
same rule, same settings, same numbers under the same names.

Copied from ``Protocols/Analysis/microglia_cosmic_ray_removal.py`` at
``2026-08-20-one-outlier-rule``, which replaced the matched-line method this
package carried until then. That one is in ``superseded/matched_line.py``,
where nothing calls it and old run records can still replay it.
"""

from __future__ import annotations

from .clean import cleaned_path, default_output_dir, remove_cosmic_rays
from .stack import clean_stack_in_place
from .rule import (
    COSMIC_STAGE,
    MAX_PROFILE,
    METHOD_VERSION,
    REFERENCE_FRAMES,
    Settings,
    band_width,
    censored,
    combined_z,
    component_span,
    fit_tail,
    full_scale,
    line_through,
    load_exclusion,
    match_track,
    measure_noise,
    outlier_mask,
    predict_tail,
    principal_axis,
    reference_plane,
    reference_window,
    robust_noise,
    score_line,
    side_profile,
    track_band,
    track_components,
    validate,
    z_image,
)

__all__ = [
    "METHOD_VERSION",
    "COSMIC_STAGE",
    "MAX_PROFILE",
    "REFERENCE_FRAMES",
    "Settings",
    "remove_cosmic_rays",
    "clean_stack_in_place",
    "default_output_dir",
    "cleaned_path",
    "reference_window",
    "reference_plane",
    "robust_noise",
    "measure_noise",
    "full_scale",
    "z_image",
    "combined_z",
    "outlier_mask",
    "censored",
    "component_span",
    "track_components",
    "principal_axis",
    "line_through",
    "score_line",
    "band_width",
    "match_track",
    "track_band",
    "side_profile",
    "fit_tail",
    "predict_tail",
    "load_exclusion",
    "validate",
]
