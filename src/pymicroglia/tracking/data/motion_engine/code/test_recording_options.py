"""Behavior checks for sparse recordings and configurable leading-frame removal."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import sys
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import tifffile

from common import Config
from observations import select_anchor
import accepted_history as history


class RecordingOptionsTests(unittest.TestCase):
    def test_numbered_export_uses_actual_movie_length(self):
        import build_numbered_outlines as export
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = np.zeros((7,24,24),np.uint16);labels[:,8:14,8:14] = 1
            raw = labels*100
            tifffile.imwrite(root/'labels.tif',labels,photometric='minisblack')
            tifffile.imwrite(root/'raw.tif',raw,photometric='minisblack')
            (root/'batch_manifest.json').write_text(json.dumps(dict(records=[dict(
                stem='fixture',registered_input=str(root/'raw.tif'),
                authoritative_final_labels=str(root/'labels.tif'),
                final_labels_sha256=export.sha256(root/'labels.tif'))])))
            with patch.object(sys,'argv',['build_numbered_outlines.py','--result-root',str(root)]), \
                 patch.object(export,'make_preview') as preview:
                export.main()
            saved=tifffile.imread(root/'numbered outlines/fixture_numbered_outlines.tif')
            self.assertEqual(saved.shape,(7,24,24,3))
            self.assertIn('retained frame 4',preview.call_args.args[0][0][0])

    def test_detector_alignment_keeps_all_frames_unless_requested(self):
        import raw_physical_hypotheses as detector
        raw = np.arange(7*8*8, dtype=np.uint16).reshape(7,8,8)
        with patch.object(detector.tifffile, 'imread', return_value=raw):
            np.testing.assert_array_equal(detector.aligned_raw({'raw_path':'fixture'}), raw)
            np.testing.assert_array_equal(detector.aligned_raw({'raw_path':'fixture','source_frame_offset':2}), raw[2:])

    def test_review_supports_short_movies_and_failed_proxy_scores(self):
        import continuity_review
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = np.arange(7*24*24,dtype=np.uint16).reshape(7,24,24)+1
            labels = np.zeros_like(raw)
            labels[:,8:14,8:14] = 1
            candidate = labels.copy();candidate[-1,8:14,8:14] = 2
            for name,array in [('raw',raw),('before',labels),('after',candidate)]:
                tifffile.imwrite(root/(name+'.tif'),array,photometric='minisblack')
            (root/'score').mkdir();(root/'qc').mkdir();(root/'out').mkdir()
            (root/'score/metrics.json').write_text(json.dumps(dict(pooled_confidence_delta=-1,
                group_mean_confidence_delta=-1,identity_set_exact=False,
                per_frame_existing_identity_preserved=False,
                assigned_unclaimed_disjoint=True,zero_signal_additions=0)))
            result = continuity_review.run(root/'score',dict(
                baseline_labels_path=str(root/'before.tif'),candidate_labels_path=str(root/'after.tif'),
                raw_path=str(root/'raw.tif'),attempt_id='synthetic_fixture'),
                SimpleNamespace(qc=root/'qc',out=root/'out'))
            self.assertEqual(result['summary']['frames'],7)
            self.assertEqual(len(result['summary']['panels']),3)
            self.assertEqual(result['summary']['changed_pixels'],36)
            self.assertTrue(result['outputs']['full_field_review'].is_file())

    def test_config_defaults_to_all_frames_and_validates_offset(self):
        self.assertEqual(Config({}, Path('example.json')).source_frame_offset, 0)
        self.assertEqual(Config({'source_frame_offset': 2}, Path('current.json')).source_frame_offset, 2)
        for bad in (-1, 1.5, True, '2'):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                Config({'source_frame_offset': bad}, Path('bad.json')).source_frame_offset

    def test_sparse_populations_are_trackable(self):
        for count in (1, 2, 4):
            with self.subTest(count=count):
                table = pd.DataFrame([
                    dict(t=t, y=3, x=3+3*i, two_sided_green=True,
                         source_kind='green', border_contact=False)
                    for t in range(9) for i in range(count)])
                scores, anchor = select_anchor(np.zeros((9, 16, 16)), table)
                self.assertEqual(anchor.observations, count)
                self.assertTrue(0 <= anchor.t < 9)

    def test_empty_central_window_uses_observed_edge_frame(self):
        table = pd.DataFrame([dict(t=0, y=3, x=3, two_sided_green=False,
                                  source_kind='green', border_contact=False)])
        _, anchor = select_anchor(np.zeros((21, 8, 8)), table)
        self.assertEqual(anchor.t, 0)

    def test_no_detections_gives_explicit_data_error(self):
        with self.assertRaisesRegex(ValueError, 'No detected bodies'):
            select_anchor(np.zeros((9, 8, 8)), pd.DataFrame())

    def test_actual_history_boundary_aligns_zero_two_and_three_frame_offsets(self):
        class BoundaryReached(Exception):
            pass
        labels = np.arange(8, dtype=np.uint16)[:, None, None]+1
        raw = labels*10
        lag = np.arange(7, dtype=np.float32)[:, None, None]
        sources = dict(registered_raw='raw', lag_float='lag', neutral_tracks='neutral', motion_composite='motion')
        arrays = dict(raw=raw, lag=lag, neutral=lag, motion=lag)
        inputs = Mock()
        inputs.required.return_value = (labels, labels, labels, pd.DataFrame(), set())
        for offset in (0, 2, 3):
            observed = []
            def boundary(candidate, aligned_raw, aligned_lag, stem, output, received_offset, *, frame_interval_min):
                observed.append(received_offset)
                self.assertEqual(frame_interval_min, 30)
                np.testing.assert_array_equal(candidate, labels[offset:])
                np.testing.assert_array_equal(aligned_raw, raw[offset:])
                np.testing.assert_array_equal(aligned_lag, lag[offset:])
                raise BoundaryReached()
            cfg = Config(dict(frame_interval_min=30, source_frame_offset=offset), Path('fixture.json'))
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as folder:
                with patch.object(history, 'load_stack', side_effect=lambda key: arrays[key]), \
                     patch.object(history, '_confidence_layered_retracking', return_value=labels.copy()), \
                     patch.object(history, '_contact_tenure', side_effect=lambda a,*args:a), \
                     patch.object(history, '_host_capture_correction', side_effect=lambda a,*args:a), \
                     patch.object(history, '_save'), patch.object(history, '_ownership_quarantine', side_effect=boundary):
                    with self.assertRaises(BoundaryReached):
                        history.run_accepted_history(cfg, 'fixture', 'run', sources, labels,
                            labels, pd.DataFrame(), 0, Path(folder), history_inputs=inputs)
                self.assertEqual(observed, [offset])

    def test_dataset_two_preserves_original_nested_settings(self):
        for name, original in history.ACCEPTED_PARAMETERS['issue023'].items():
            self.assertEqual(history._general(name, 2), original)
        for issue, name in [('issue010','persistence_profile'), ('issue010','reclaim_gaps'),
                            ('issue011','merge_split_reconcile')]:
            self.assertEqual(history._params(issue, name, 2), history.ACCEPTED_PARAMETERS[issue][name])
        values = history._general('R09_generalise_reciprocal_lineages', 0)
        self.assertEqual(values['source_frame_offset'], 0)
        self.assertEqual(values['handoff_rule']['source_frame_offset'], 0)
        self.assertEqual(history.ACCEPTED_PARAMETERS['issue023']['R09_generalise_reciprocal_lineages']['source_frame_offset'], 2)

    def test_final_owner_transport_receives_recording_offset(self):
        scripts = Path(__file__).resolve().parents[1]/'issues/ISSUE-009-completed-feedback-rule-optimization/analysis/completed_feedback_tuning/scripts'
        sys.path.insert(0, str(scripts))
        import final_ownership_composition as final
        received = []
        @contextmanager
        def method(spec):
            def execute(labels, original, points, traces, *, source_frame_offset):
                received.append(source_frame_offset)
                return labels, pd.DataFrame()
            yield execute
        a = np.zeros((3, 2, 2), np.uint16)
        evidence = {'gap': dict(original_labels=a, points=pd.DataFrame(), traces=[])}
        with patch.object(final, 'frozen_method', method):
            list(final.apply(a, a, evidence, [dict(stage='gap', kind='gap')]))
            list(final.apply(a, a, evidence, [dict(stage='gap', kind='gap')], source_frame_offset=2))
        self.assertEqual(received, [0, 2])


if __name__ == '__main__':
    unittest.main()
