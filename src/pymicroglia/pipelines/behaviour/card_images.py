"""Exact-observation image crops for saved state representatives."""
from pymicroglia._results import read_document
from pathlib import Path
import json
import tempfile

import numpy as np

from pymicroglia.pipelines._contracts import content_id
from pymicroglia.pipelines.rhythm.images import IMAGE_DEFAULTS, prepare
from pymicroglia.pipelines._screening import _write_json


def prepare_examples(context, examples, settings, destination):
    """Reuse verified source pixels, refusing a nearest-frame substitution.

    The image helper may propose a nearest recorded frame; a state example is
    accepted only when its selected observation frame and physical time match.
    Original crops, masks and display arrays are frozen separately from science.
    """
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    arrays, inventory, sources = {}, [], {}
    for example in examples:
        identity = {name: example[name] for name in ['source_run', 'movie', 'identity']}
        if identity['source_run'] != context.request.inputs.source_run:
            raise ValueError('State image example belongs to a different source run')
        row = {**example, 'image_status': 'unavailable', 'image_reason': 'No exact recorded image available', 'tiles': []}
        inventory.append(row)
        if not settings['state_card_images']:
            row.update(image_status='not_requested', image_reason='Representative imagery disabled'); continue
        options = {**IMAGE_DEFAULTS, **{key: settings[key] for key in IMAGE_DEFAULTS if key in settings},
            'images': 1, 'image_hours': [example['hours']]}
        with tempfile.TemporaryDirectory(prefix='state-card-image-') as temporary:
            archive, metadata, declared_sources = prepare(context, [identity], options, Path(temporary))
            for name, source in declared_sources.items():
                sources[content_id({'name': name, 'path': str(source)})+'_'+name] = source
            records = read_document(metadata)['cells']
            if len(records) != 1: raise ValueError('An exact state example resolved more than one cell image record')
            image = records[0]
            if any(image.get(key) != identity[key] for key in identity): raise ValueError('State imagery belongs to a different full cell identity')
            if image['status'] != 'available':
                row.update(image_status=image['status'], image_reason=image['reason']); continue
            tiles = image['tiles']
            if len(tiles) != 1 or tiles[0]['frame_index'] != example['frame_index'] or tiles[0]['hours'] != example['hours']:
                row['image_reason'] = 'Available image frame does not exactly match the selected state observation; nearest-frame substitution refused'; continue
            if not tiles[0]['cell_present']:
                row['image_reason'] = 'The selected cell is absent from the exact recorded label frame'; continue
            prefix = 'example_'+example['example_id']
            with np.load(archive, allow_pickle=False) as saved:
                for quantity in ['raw', 'mask', 'display']:
                    arrays[prefix+'_'+quantity] = saved[image['archive_key']+'_'+quantity].copy()
            row.update(image_status='available', image_reason='', archive_key=prefix, tiles=tiles,
                crop_box=image['crop_box'], display_settings=image['settings'], image_sources=image['image_sources'])
    archive, metadata = destination/'state_example_tiles.npz', destination/'state_example_images.json'
    np.savez_compressed(archive, **arrays)
    metadata = _write_json(metadata, {'schema_version': 1, 'display_only': True, 'lookup_rule': 'Exact selected observation frame and original physical hours; no nearest-frame substitution',
        'examples': inventory})
    return archive, metadata, sources
