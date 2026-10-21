"""Motion-first fluorescence masks and original Red/Green measurements."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import json
import math
import shutil
import tempfile

METHOD_VERSION = '2026-10-14-native-red-green-motion-first-v2'
PIPELINE = 'native_red_green'
STAGES = ('register', 'red_minus_green', 'motion_inputs', 'motion', 'optional_refinement', 'native_masks', 'measure', 'display')


def validate_options(*, phase=None, red=None, green=None, registration_manifest=None,
                     registration_method='ripr', ripr_recipe=None, outline_mode='expanded',
                     expansion_px=3., background_inner_px=4., background_outer_px=12.,
                     min_background_px=25, background_statistic='mean', frame_interval_h=.5,
                     mask_weights=None, recordings=None, post_tracking_refinement=False,
                     signal_isolation='raw_difference', isolation_options=None, **unused):
    from auto_organotypic.motion_runtime import RECIPES
    if outline_mode not in {'original', 'expanded'}:
        raise ValueError('Choose original or expanded measurement outlines')
    if registration_method not in {'ripr', 'structural_rigid'}:
        raise ValueError('Choose packaged RIPR or structural rigid registration')
    if not math.isfinite(expansion_px) or expansion_px < 0:
        raise ValueError('Expansion must be finite and nonnegative')
    if not all(math.isfinite(v) for v in (background_inner_px, background_outer_px)) or not 0 <= background_inner_px < background_outer_px:
        raise ValueError('Background rings require 0 <= inner < outer')
    if isinstance(min_background_px, bool) or int(min_background_px) != min_background_px or min_background_px < 1:
        raise ValueError('Minimum background pixels must be a positive integer')
    if background_statistic not in {'mean', 'median'}:
        raise ValueError('Background statistic must be mean or median')
    if not math.isfinite(frame_interval_h) or frame_interval_h <= 0:
        raise ValueError('Frame interval must be finite and positive')
    if post_tracking_refinement and not mask_weights:
        raise ValueError('Select model weights only when post-Motion refinement is enabled')
    if signal_isolation not in {'raw_difference','imagej_macro'}:
        raise ValueError('Choose raw_difference or imagej_macro signal isolation')
    if signal_isolation=='raw_difference' and isolation_options:
        raise ValueError('Isolation options require the ImageJ macro recipe')
    entries = recordings if recordings is not None else [dict(phase=phase, red=red, green=green,
                                                             registration_manifest=registration_manifest)]
    if not entries or not isinstance(entries, (list, tuple)):
        raise ValueError('Recordings must be a nonempty list of input mappings')
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError('Each recording must be an input mapping')
        if entry.get('registration_manifest'):
            if any(entry.get(name) for name in ('phase', 'red', 'green')):
                raise ValueError('Choose a retained registration manifest or three original channels')
        else:
            if not all(entry.get(name) for name in ('phase', 'red', 'green')):
                raise ValueError('Supply original Phase, Red and Green TIFFs')
            if registration_method == 'ripr' and ripr_recipe not in RECIPES:
                raise ValueError('Choose an explicit packaged RIPR recipe')
    return entries


def _one(entry, folder, settings):
    import numpy as np
    import tifffile
    from auto_organotypic.native_channel_registration import export, digest, anticipated_download_bytes
    from ..tracking.fluorescence import prepare, foreground, refine
    from ..tracking import run as track
    from ..tracking.native_masks import project
    from ..measure.expanded_channels import measure, blocked_foreground
    folder.mkdir(parents=True, exist_ok=True)
    if entry.get('registration_manifest'):
        manifest_path = Path(entry['registration_manifest']).resolve()
        registration = json.loads(manifest_path.read_text(encoding='utf8'))
    else:
        registration = export(entry['phase'], {'Red': entry['red'], 'Green': entry['green']},
            output_dir=folder/'registration', method=settings['registration_method'],
            ripr_recipe=settings['ripr_recipe'], workers=settings['threads'],
            frame_interval_h=settings['frame_interval_h'], frame_times=entry.get('frame_times'),
            registration_options=settings['registration_options'])
        manifest_path = folder/'registration/manifest.json'
    if registration.get('source_processed') is not False:
        raise ValueError('Retained registration must explicitly retain original quantitative pixels')
    roles = {row['role']: row for row in registration['outputs']}
    required = {'tracking_proxy', 'validity_canvas', 'red_original', 'green_original', 'source_to_canvas', 'frame_clock'}
    if required - set(roles):
        raise ValueError('Registration manifest lacks required original-channel or pose outputs')
    pixels = math.prod(registration['canvas_shape'])
    free = min(shutil.disk_usage(folder).free, shutil.disk_usage(tempfile.gettempdir()).free)
    download_bytes = anticipated_download_bytes(Path(row['path']) for row in roles.values())
    if free < pixels*26+2*1024**3+download_bytes:
        raise OSError('Insufficient disk for complete mask and Motion preparation, including scratch')
    for row in roles.values():
        if digest(row['path']) != row['sha256']:
            raise ValueError('Retained registration output changed: '+row['role'])
    from ..guards import require_measurement
    for role in ('red_original','green_original'):
        require_measurement(roles[role]['path'])
    model_hash = digest(settings['mask_weights']) if settings['post_tracking_refinement'] else None
    cells_path = folder/'cells.tif'
    prepared = prepare(registration, folder/'motion_inputs',stem=folder.name,
                       frame_interval_min=settings['frame_interval_h']*60,
                       signal_isolation=settings.get('signal_isolation','raw_difference'),
                       isolation_options=settings.get('isolation_options'))
    handoff = folder/'motion_inputs.json'
    handoff.write_text(json.dumps({'stems':[folder.name], 'prepared':{folder.name:prepared}}, indent=2))
    options = dict(settings['tracker_options'] or {})
    options.setdefault('bounded_memory', True)
    options.setdefault('resume', bool(options['bounded_memory']))
    tracking = track(handoff, folder/'tracking', tracker_options=options).as_dict()
    (folder/'tracking_result.json').write_text(json.dumps(tracking, indent=2))
    offset = tracking['source_frame_offset']
    if offset != 0:
        raise ValueError('Full-recording fluorescence Motion must retain all acquired frames')
    foreground(tracking['labels'],cells_path,unclaimed=tracking.get('unclaimed'))
    measurement_labels=tracking['labels']
    if settings['post_tracking_refinement']:
        measurement_labels=refine(prepared['tracking_signal'],tracking['labels'],folder/'refined_labels.tif',
            weights=settings['mask_weights'],method=settings['refinement_method'],
            options=settings['mask_options'],threads=settings['threads'])
    inputs = {'labels':{'path':measurement_labels}, 'foreground':{'path':str(cells_path), 'source_frame_offset':0}}
    for role in ('unclaimed', 'provenance'):
        if tracking.get(role):
            inputs[role] = {'path':tracking[role]}
    projection = project(inputs, roles['source_to_canvas']['path'], native_shape=registration['native_shape'][1:],
                         output_dir=folder/'native_masks', source_frame_offset=offset)
    native = projection['outputs']
    blocked = blocked_foreground(native['labels']['path'], native['foreground']['path'],
        validity=roles.get('native_validity',{}).get('path'),
        source_frame_offset=offset, unclaimed=native.get('unclaimed',{}).get('path'), output=folder/'blocked.tif')
    outline_options = dict(margin_px=settings['expansion_px'] if settings['outline_mode']=='expanded' else 0.,
        background_inner_px=settings['background_inner_px'], background_outer_px=settings['background_outer_px'],
        min_background_px=settings['min_background_px'], background_statistic=settings['background_statistic'])
    report = measure({'Red':roles['red_original']['path'], 'Green':roles['green_original']['path']},
        native['labels']['path'], output_dir=folder/'measurement', blocked=blocked, options=outline_options,
        source_frame_offset=offset, frame_interval_h=settings['frame_interval_h'], frame_times=roles['frame_clock']['path'],
        claim=settings['claim'])
    if model_hash is not None and digest(settings['mask_weights']) != model_hash:
        raise ValueError('Mask model checkpoint changed during the workflow')
    return dict(folder=str(folder), mask_weights_sha256=model_hash,
                registration_manifest=str(manifest_path), registration_sha256=digest(manifest_path),
                measurement_dir=str(folder/'measurement'), labels=native['labels']['path'], blocked=str(blocked),
                channels={name:roles[name.lower()+'_original']['path'] for name in ('Red','Green')},
                report=report, frames=registration['frames'], offset=offset, outline_options=outline_options)


def run(*, output_dir, mask_weights=None, phase=None, red=None, green=None,
        registration_manifest=None, recordings: Sequence[Mapping] | None=None,
        registration_method: str='ripr', ripr_recipe: str | None=None,
        registration_options: Mapping | None=None, frame_times=None, frame_interval_h: float=.5,
        outline_mode: str='expanded', expansion_px: float=3., background_subtraction: bool=True,
        background_inner_px: float=4., background_outer_px: float=12., min_background_px: int=25,
        background_statistic: str='mean', mask_options: Mapping | None=None,
        post_tracking_refinement: bool=False, refinement_method: str='single_frame',
        signal_isolation: str='raw_difference', isolation_options: Mapping | None=None,
        tracker_options: Mapping | None=None, threads: int=4,
        comparison_videos: bool=False, video_identities: Sequence[int]=(),
        video_options: Mapping | None=None, minimum_available_gib: float=4.,
        maximum_private_gib: float=4., claim: str=''):
    """Run all acquired frames over the entire camera; keep both original signal channels.

    Supply original single-channel TIFFs or a hash-pinned full-camera registration
    manifest. ``recordings`` runs several mappings serially in one saved workspace.
    Existing Motion identities compete before any expansion. The three totals
    are always retained; background_subtraction selects the preferred readout.
    Comparison videos use the same saved apertures and values.
    """
    settings = dict(locals())
    import sys
    import time
    validate_options(**settings)
    if not math.isfinite(minimum_available_gib) or minimum_available_gib < 1:
        raise ValueError('Keep at least 1 GiB available RAM for this workflow')
    if not math.isfinite(maximum_private_gib) or not 0 < maximum_private_gib <= 4:
        raise ValueError('The analysis process memory cap must be greater than zero and at most 4 GiB')
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh output directory; retained runs are never overwritten')
    logs = out.parent/'.native-run-logs';logs.mkdir(parents=True, exist_ok=True)
    token = str(time.time_ns())
    request = logs/(token+'_request.json')
    receipt = logs/(token+'_resources.json')
    request.write_text(json.dumps(settings, default=str), encoding='utf8')
    with (logs/(token+'.log')).open('w', encoding='utf8') as log:
        # The guarded command inherits redirected OS descriptors through Popen,
        # rather than running a memory-intensive workflow in the desktop server.
        import subprocess
        args = [sys.executable, '-m', 'auto_organotypic.resource_guard', '--receipt', str(receipt),
                '--reserve-gib', str(minimum_available_gib), '--limit-gib', str(maximum_private_gib),
                '--start-headroom-gib', str(minimum_available_gib), '--stable-seconds', '0',
                '--', sys.executable, '-m', __name__, str(request)]
        code = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        _cleanup_failed_scratch(out, receipt, logs/(token+'_cleanup.json'))
        raise RuntimeError(f'Workflow stopped; original inputs and failure evidence retained in {logs/(token+".log")}')
    report = json.loads((out/'manifest.json').read_text(encoding='utf8'))
    report['resource_receipt'] = str(receipt)
    (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


def _cleanup_failed_scratch(output, receipt, cleanup_record):
    """Retire only this stopped job's recorded temporary mappings, keeping its evidence."""
    import psutil
    if not receipt.is_file():
        return
    resource = json.loads(receipt.read_text())
    if resource.get('pid') and psutil.pid_exists(resource['pid']):
        return
    base = Path(tempfile.gettempdir()).resolve()
    removed = []
    markers=list(output.glob('*/runtime_storage.json'))+list(output.glob('*/motion_inputs/runtime_storage.json'))
    for marker in markers:
        saved = json.loads(marker.read_text())
        if saved.get('owner_pid') != resource.get('pid'):
            continue
        folder = Path(saved['temporary_directory']).resolve()
        if not folder.exists():
            continue
        if not folder.is_relative_to(base) or not folder.name.startswith(('pm-native-red-','fluorescence-motion-')):
            raise ValueError('Recorded scratch is outside this workflow temporary directory')
        if folder.is_symlink() or any(path.is_symlink() for path in folder.rglob('*')):
            raise ValueError('Leave linked scratch paths untouched')
        size = sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
        shutil.rmtree(folder)
        removed.append({'temporary_directory':str(folder), 'removed_bytes':size})
    cleanup_record.write_text(json.dumps({'removed':removed, 'retained':'Original sources, generated stage outputs and failure logs'},indent=2))


def _execute(settings):
    entries = validate_options(**settings)
    threads = settings['threads']
    if isinstance(threads, bool) or not 1 <= threads <= 16 or int(threads) != threads:
        raise ValueError('Threads must be an integer between 1 and 16')
    names = [str(entry.get('name') or f'recording_{i+1}') for i, entry in enumerate(entries)]
    import re
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
        raise ValueError('Recording names must be unique and contain letters, digits, underscores or hyphens')
    out = Path(settings['output_dir']).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh output directory; retained runs are never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, entry in zip(names, entries):
        frame_times = settings['frame_times']
        entry = {**entry, **({'frame_times':frame_times} if frame_times and not entry.get('frame_times') else {})}
        results.append(_one(entry, out/name, settings))
    from .native_measurement_workspace import write_workspace, render_comparisons
    preferred = 'corrected_expanded_integrated_density' if settings['background_subtraction'] else 'expanded_integrated_density'
    workspace = write_workspace(results, out/'workspace', preferred_metric=preferred)
    videos = render_comparisons(results, out/'videos', identities=settings['video_identities'],
        options=settings['video_options']) if settings['comparison_videos'] else []
    from auto_organotypic.native_channel_registration import digest
    report = dict(pipeline=PIPELINE, method_version=METHOD_VERSION, settings=settings,
        measurement_run=str(workspace), recordings=results, outputs={'workspace':str(workspace), 'videos':videos},
        source_code_sha256=digest(__file__), validation='Opt-in workflow; no full-recording comparison trial performed')
    (out/'manifest.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf8')
    return report


if __name__ == '__main__':
    import sys
    _execute(json.loads(Path(sys.argv[1]).read_text(encoding='utf8')))
