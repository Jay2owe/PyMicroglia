"""Lifecycle decisions follow original tracking geometry, never display refinement."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

from pymicroglia.measure.context import MeasurementContext, Scale
from pymicroglia.measure.modules.lifecycle import measure as lifecycle
from pymicroglia.ui.events import load_events
from pymicroglia.ui.exploration_context import load_track_timeline
from pymicroglia.ui.exploration_data import recording, table_path
from pymicroglia.ui.exploration_lifecycle import calculate_lifecycle_review
from pymicroglia.ui.exploration_results import saved_results


@pytest.fixture
def original_motion_run(tmp_path):
    original = np.zeros((5,24,24),dtype=np.uint16)
    original[0,6:14,6:14] = 19
    original[1:,6:14,6:10] = 19
    original[1:,6:14,10:14] = 26
    refined = original.copy()
    refined[refined==26] = 0
    refined[1:,9,11] = 26  # Same identity continuity, sharply changed display area.
    source = tmp_path/'motion.tif'
    tifffile.imwrite(source,original,photometric='minisblack')
    tifffile.imwrite(tmp_path/'refined.tif',refined,photometric='minisblack')
    tifffile.imwrite(tmp_path/'raw.tif',np.ones((7,24,24),dtype=np.uint16),photometric='minisblack')
    unclaimed = tmp_path/'unclaimed.tif'
    tifffile.imwrite(unclaimed,np.zeros_like(original),photometric='minisblack')
    folder = tmp_path/'measure/one'
    folder.mkdir(parents=True)
    with (folder/'cell_frame.csv').open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['identity','frame_index','hours'])
        for frame in range(5):
            for identity in np.unique(refined[frame]):
                if identity:writer.writerow([int(identity),frame,frame+2])
    (tmp_path/'manifest.json').write_text(json.dumps({'movies':[{'stem':'one',
        'spec':{'labels':'refined.tif','raw':'raw.tif','source_frame_offset':2},
        'provenance':{'inputs':{'labels':{'path':'refined.tif'},'raw':{'path':'raw.tif'}},
                      'scale':{'minutes_per_frame':60}}}]}))
    return tmp_path, source, unclaimed, hashlib.sha256(source.read_bytes()).hexdigest(), original, refined


def test_lifecycle_review_uses_original_geometry_and_coordinates_every_view(original_motion_run):
    root, source, unclaimed, checksum, original, refined = original_motion_run
    protected = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    # Classifying the refined outline would lose this candidate; the attachment must not.
    result = lifecycle(MeasurementContext('one',refined,np.ones_like(refined),Scale(60),[19,26],source_frame_offset=2))
    assert result['lifecycle_events'].query('identity==26 and event=="appeared"').shape[0]==1
    attached = calculate_lifecycle_review(root,'one',tracking_labels=source,
                                          expected_tracking_sha256=checksum,unclaimed=unclaimed)
    assert not attached['reused']
    catalog = load_events(root,'one')
    birth = next(row for row in catalog['rows'] if row['event']=='born')
    assert birth['identity']==26 and birth['partners']==[19] and birth['status']=='candidate'
    assert birth['source_frame']==4 and birth['hours']==3
    assert any(row['event']=='divided' and row['identity']==19 for row in catalog['rows'])
    assert catalog['lifecycle_saved'] and catalog['identities']==[19,26]
    timeline = load_track_timeline(root,'one')
    assert next(row for row in timeline['cells'] if row['identity']==26)['start_event']=='born'
    _, _, movie = recording(root,'one')
    assert any(row['label']=='born (candidate)' for row in saved_results(root,movie)['events']['rows'])
    from pymicroglia.figure_tables.film_action import follow
    film = follow(root,stem='one',identities=[26],events=['born'],span='event',
                  event_frames=[1],event_hours=1,dry_run=True)['clips']
    assert len(film)==1 and film[0]['frames']==[0,1,2]
    assert film[0]['hours']==[2,3,4] and film[0]['event']=='born'
    assert all(p.read_bytes()==value for p,value in protected.items())
    repeated = calculate_lifecycle_review(root,'one',tracking_labels=source,
                                          expected_tracking_sha256=checksum,unclaimed=unclaimed)
    assert repeated['reused'] and repeated['review_run']==attached['review_run']
    replay = subprocess.run([sys.executable,attached['script']],capture_output=True,text=True,timeout=45)
    assert replay.returncode==0,replay.stderr
    replay_root = Path(attached['script']).parent/'replayed/lifecycle_review'
    for name in ('lifecycle_events','lifecycle_cells'):
        actual = table_path(root,movie,name)
        _, _, replay_movie = recording(replay_root,'one')
        assert actual.read_bytes()==table_path(replay_root,replay_movie,name).read_bytes()
    # Altering an original input cannot leave old classifications looking current.
    tifffile.imwrite(source,original+np.uint16(1),photometric='minisblack')
    with pytest.raises(ValueError,match='tracking inputs changed'):
        load_events(root,'one')


def test_missing_tracking_provenance_never_substitutes_refined_masks(original_motion_run):
    root, _, _, _, _, _ = original_motion_run
    with pytest.raises(ValueError,match='Declare the original'):
        calculate_lifecycle_review(root,'one')
    assert not (root/'AI_Exports').exists()


def test_refined_identity_loss_rejects_classification_before_writing(original_motion_run):
    root, source, unclaimed, checksum, _, refined = original_motion_run
    refined[1,9,11] = 0
    tifffile.imwrite(root/'refined.tif',refined,photometric='minisblack')
    with pytest.raises(ValueError,match='identities differ'):
        calculate_lifecycle_review(root,'one',tracking_labels=source,
                                   expected_tracking_sha256=checksum,unclaimed=unclaimed)
    assert not (root/'AI_Exports').exists()
