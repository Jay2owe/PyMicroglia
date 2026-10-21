"""Live, allowlisted UI schema for PyMicroglia's complete workflows."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def _kind(annotation: Any, default: Any) -> str:
    hint = str(annotation).lower()
    if "bool" in hint or isinstance(default, bool):
        return "bool"
    if "mapping" in hint or "dict" in hint or isinstance(default, dict):
        return "mapping"
    if "sequence" in hint or "list" in hint or isinstance(default, (list, tuple)):
        return "list"
    if "path" in hint or isinstance(default, Path):
        return "path"
    if "float" in hint or isinstance(default, float):
        return "float"
    if "int" in hint or isinstance(default, int):
        return "int"
    return "str"


def schema(name: str) -> dict:
    from pymicroglia.pipelines import PIPELINE_NAMES, get

    if name not in PIPELINE_NAMES:
        raise KeyError(name)
    module = get(name)
    signature = inspect.signature(module.run)
    params = []
    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        required = parameter.default is inspect.Parameter.empty
        default = None if required else parameter.default
        if isinstance(default, Path):
            default = str(default)
        elif isinstance(default, tuple):
            default = list(default)
        params.append({"name": parameter.name, "type": _kind(parameter.annotation, default),
                       "required": required, "default": default,
                       "description": "Complete workflow input" if required else "Optional workflow setting"})
    if name == 'native_red_green':
        from auto_organotypic.motion_runtime import RECIPES
        descriptions = {
            'phase': 'Original Phase TIFF used to estimate one shared camera pose.',
            'red': 'Original Red TIFF; Red minus Green supplies Motion detection, while original pixels supply Red measurements.',
            'green': 'Original Green TIFF; subtract from Red for detection and measure untouched with the same Motion outlines.',
            'registration_manifest': 'Reuse a verified full-camera export instead of supplying three channels.',
            'recordings': 'Optional batch: input mappings with a unique name and Phase/Red/Green files or registration_manifest. Run serially in one workspace.',
            'mask_weights': 'Optional U-Net checkpoint, used only for refinement after Motion tracking.',
            'post_tracking_refinement': 'Optionally refine Motion masks on the registered Red-minus-Green signal after identities are assigned.',
            'signal_isolation': 'Choose direct subtraction or the recorded ImageJ background and contrast recipe for detection. Original-channel measurements are unchanged.',
            'isolation_options': 'ImageJ recipe settings: rolling_radius (10 pixels) and saturation (0.35 percent).',
            'refinement_method': 'Independent post-Motion outline provider; single_frame uses the single-frame U-Net.',
            'registration_method': 'Choose the packaged Relative-Intensity Pattern Registration or the separate structural feature matcher.',
            'ripr_recipe': 'Select the packaged recipe explicitly when using Relative-Intensity Pattern Registration.',
            'outline_mode': 'Original uses tracked pixels; expanded adds nearby pixels without sharing ownership with another cell.',
            'expansion_px': 'Maximum expansion in original camera pixels; other cells and untracked foreground limit growth.',
            'background_subtraction': 'Use expanded total minus local background as the preferred readout. All three comparison totals remain saved.',
            'comparison_videos': 'Create cropped original Red/Green cell movies with the same measured outlines and three traces below.',
            'video_identities': 'Cell IDs to film in each recording; leave empty to film all measured cells.',
            'frame_times': 'Optional CSV with source_frame and hours. Missing acquisitions remain gaps.',
            'frame_interval_h': 'Hours between frames when no acquisition-time CSV is supplied.',
            'minimum_available_gib': 'Wait for this much available RAM; stop below this floor (minimum 1 GiB).',
            'maximum_private_gib': 'Hard memory cap per analysis process (at most 4 GiB).',
        }
        primary = ['output_dir','phase','red','green','registration_manifest',
                   'registration_method','ripr_recipe','outline_mode','expansion_px',
                   'background_subtraction','signal_isolation','post_tracking_refinement','mask_weights','comparison_videos']
        for row in params:
            row['description'] = descriptions.get(row['name'], row['description'])
            if row['name'] in primary:
                row['grid_primary_order'] = primary.index(row['name'])
            if row['name']=='registration_method':
                row.update(choices=['ripr','structural_rigid'], choice_labels={
                    'ripr':'Auto-Organotypic: Relative-Intensity Pattern Registration',
                    'structural_rigid':'Structural features: rigid rotation and translation'})
            if row['name']=='ripr_recipe':
                row.update(choices=list(RECIPES), choice_labels={key:value[2] for key,value in RECIPES.items()})
            if row['name']=='outline_mode':
                row.update(choices=['original','expanded'], choice_labels={
                    'original':'Original tracked outline','expanded':'Expanded outline, exclusive cell ownership'})
            if row['name']=='signal_isolation':
                row.update(choices=['raw_difference','imagej_macro'],choice_labels={
                    'raw_difference':'Direct Red minus Green',
                    'imagej_macro':'ImageJ background removal and normalized channel subtraction'})
            if row['name']=='background_statistic':
                row.update(choices=['mean','median'])
    return {"name": name, "pipeline": True,
            "summary": (module.__doc__ or "").strip().split("\n")[0],
            "params": params, "pending": False, "claim_required": False,
            "display_only": False,
            "method_version": getattr(module, "METHOD_VERSION", "")}


def validate(name: str, params: dict) -> dict:
    from pymicroglia.pipelines import get

    spec = schema(name)
    if not isinstance(params, dict):
        return {"ok": False, "message": "Workflow settings must be a JSON object"}
    try:
        inspect.signature(get(name).run).bind(**params)
        if name == 'native_red_green':
            get(name).validate_options(**params)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "message": str(exc), "problems": [str(exc)]}
    return {"ok": True, "pipeline": name, "problems": []}
