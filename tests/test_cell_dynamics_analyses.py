"""Numerical parity with the approved views, independent of any HTML client."""
import numpy as np
import pandas as pd
import pytest
from test_cell_review_data import recording
from pymicroglia.measure.cell_signal import analyse as signal
from pymicroglia.measure.mask_changes import analyse as changes, compare
from pymicroglia.figure_tables.outline_history import analyse as outlines
from pymicroglia.figure_tables.recording_barcode import matrix
from pymicroglia.figure_tables.cell_cards import analyse as cards, frame_plan
from pymicroglia.figure_tables.cell_review_data import prepare as approved


def test_original_pixel_signals_match_the_approved_view_without_encoding_images(recording):
    raw, labels, pixels, masks = recording
    table = signal(raw, labels, source_frame_offset=1, frame_interval_h=.25)
    prior = approved(raw, labels, source_frame_offset=1, frame_interval_h=.25)
    assert sorted(table.identity.unique()) == [19, 26]
    assert table.source_frame_index.max() == 5
    for identity, group in table.groupby("identity", sort=False):
        cell = prior["cells"][str(identity)]
        for column, old in [("integrated_density", "signal_raw"),
                            ("integrated_density_mean3", "signal_mean3"), ("area_px", "area")]:
            np.testing.assert_allclose(group[column], np.asarray(cell[old], float), equal_nan=True)
    assert table.loc[table.source_frame_index.eq(0), "mask_present"].eq(False).all()
    assert table.loc[(table.identity == 19) & (table.source_frame_index == 4), "integrated_density"].isna().all()


def test_translation_removal_and_missing_pairs_match_approved_semantics(recording):
    _, _, _, masks = recording
    delta = compare(masks[0] == 19, masks[1] == 19, coordinates="cell")
    assert delta["gained"] == [] and delta["lost"] == [] and delta["retained_px"] == 6
    original = compare(masks[0] == 19, masks[1] == 19, coordinates="recording")
    assert len(original["gained"]) == len(original["lost"]) == 2
    protrusion = compare(masks[1] == 19, masks[2] == 19, coordinates="recording")
    assert protrusion["gained"] == [(4, 2)] and not protrusion["lost"]
    table = changes(masks, source_frame_offset=1, frame_interval_h=.25)["summary"]
    absent = table.loc[(table.identity == 19) & table.source_frame_index.isin([1, 4, 5])]
    assert not absent.comparison_available.any() and absent.gained_px.isna().all()
    assert table.loc[(table.identity == 26), "gained_px"].isna().all()


def test_outline_slices_preserve_components_native_times_and_gaps(recording):
    _, _, _, masks = recording
    history = outlines(masks, identity=19, source_frame_offset=1, frame_interval_h=.25)
    assert sorted(history["vertices"].source_frame_index.unique()) == [1, 2, 3, 5]
    assert not history["presence"].query("source_frame_index == 4").mask_present.any()
    np.testing.assert_allclose(history["vertices"].hours, history["vertices"].source_frame_index * .25)
    separated = np.zeros_like(masks)
    separated[0, 2, 2] = 26
    separated[0, 9, 11] = 26
    disconnected = outlines(separated, identity=26, coordinates="recording")
    assert disconnected["vertices"].component.nunique() == 2


def test_barcode_scaling_keeps_values_and_missingness_independent(recording):
    raw, labels, _, _ = recording
    table = signal(raw, labels, source_frame_offset=1)
    local = matrix(table, metric="integrated_density", scale="cell")
    global_ = matrix(table, metric="integrated_density", scale="recording")
    np.testing.assert_array_equal(local["raw"], global_["raw"])
    np.testing.assert_array_equal(np.isnan(local["raw"]), np.isnan(local["display"]))
    assert local["ids"] == [19, 26] and list(local["frames"]) == list(range(6))
    assert local["table"].query("identity == 19 and source_frame_index == 4").raw_value.isna().all()
    with pytest.raises(ValueError, match="one value"):
        matrix(pd.concat([table, table.iloc[:1]]))


def test_card_plan_keeps_sharp_current_pixels_and_absent_masks(recording):
    raw, labels, pixels, masks = recording
    data = cards(raw, labels, identities=[19], source_frame_offset=1, frame_interval_h=.25)
    observed = frame_plan(data, 1)[0]
    assert observed["mask_present"] and observed["hours"] == .25
    assert observed["raw_crop"].dtype == pixels.dtype
    absent = frame_plan(data, 4)[0]
    assert not absent["mask_present"] and not absent["boundary"].any()
    assert data["table"].query("source_frame_index == 4").integrated_density.isna().all()
    # The crop contains only this source frame's values plus edge padding.
    assert set(np.unique(absent["raw_crop"])) <= set(np.unique(pixels[4])) | {0}
    np.testing.assert_array_equal(masks, recording[3])


def test_no_identity_is_invented_or_renumbered(recording):
    raw, labels, pixels, masks = recording
    for function, kwargs in [(signal, {"raw": raw, "labels": labels}),
                             (cards, {"raw": raw, "labels": labels}),
                             (changes, {"labels": labels})]:
        before = (raw.read_bytes(), labels.read_bytes())
        function(**kwargs)
        assert before == (raw.read_bytes(), labels.read_bytes())
    with pytest.raises(ValueError, match="absent"):
        outlines(masks, identity=99)
