"""Scientific data boundaries and replay for the optional cell grouping step."""

import json
from pymicroglia._results import document

import numpy as np
import pandas as pd
import pytest

from pymicroglia.clustering.cohort import (build_features, fit_cells, load_tables,
                                      run_clustering)
from pymicroglia.clustering.fingerprint import (ClusteringOptions, prepare_features,
                                       replay_fingerprints, split_cells)


def measurements(n=80):
    rng = np.random.default_rng(7)
    group = np.arange(n) % 2
    return pd.DataFrame({
        "stem": np.repeat(["movie_a", "movie_b", "movie_c", "movie_d"], n // 4),
        "identity": np.tile(np.arange(n // 4), 4),
        "subject": np.repeat(["animal_a", "animal_a", "animal_b", "animal_c"], n // 4),
        "condition": np.repeat(["control", "control", "treated", "treated"], n // 4),
        "area_px_median": 100 + 30 * group + rng.normal(0, 2, n),
        "circularity_median": .3 + .3 * group + rng.normal(0, .03, n),
        "solidity_median": .5 + .3 * group + rng.normal(0, .03, n),
        "turnover_index_median": .15 + .4 * group + rng.normal(0, .02, n),
        "signal_mean_median": 200 + 80 * group + rng.normal(0, 5, n),
        "skeleton_px_median": 20 + 30 * group + rng.normal(0, 2, n),
        "inferred_fraction_median": rng.uniform(0, 1, n),
        "secret_treatment_code": group,
    })


def test_features_keep_movie_cell_keys_and_exclude_labels_quality_and_phase():
    summary = measurements()
    frame = summary[["stem", "identity", "subject", "condition"]].copy()
    frame["frame_index"] = 0
    frame["area_px"] = summary.area_px_median
    frame["best_phase_hours"] = np.arange(len(frame))
    meta, data, families, audit, _ = build_features({"cell_summary": summary, "cell_frame": frame})
    assert len(meta) == 80
    assert not meta.duplicated(["stem", "identity"]).any()
    assert not any(x in " ".join(data.columns) for x in ("phase", "secret", "inferred", "cell_frame"))
    assert set(families.values()) >= {"morphology", "intensity", "surveillance"}
    assert any(a["reason"] == "already_in_cell_summary" for a in audit)


def test_duplicate_keys_are_errors_not_averaged():
    data = measurements()
    with pytest.raises(ValueError, match="unique"):
        build_features({"cell_summary": pd.concat([data, data.iloc[:1]])})


def test_auxiliary_channels_stay_separate():
    summary = measurements()
    tracks = pd.concat([summary[["stem", "identity"]].assign(channel=c, channel_level_median=v)
                        for c, v in [("red", 10.), ("green", 100.)]], ignore_index=True)
    _, features, _, _, _ = build_features({"cell_summary": summary, "channel_tracks": tracks})
    red = [c for c in features if '"red"' in c]
    green = [c for c in features if '"green"' in c]
    assert len(red) == len(green) == 1
    assert features[red[0]].eq(10).all()
    assert features[green[0]].eq(100).all()


def test_unknown_rhythms_are_missing_and_periods_need_support():
    summary = measurements()
    rhythms = summary[["stem", "identity"]].copy()
    rhythms = rhythms.assign(metric="area_px", period_estimation_method="mesa",
                             primary_rhythm_test="lomb", workbench_version="test",
                             detrend="linear", detrend_window_hours=24,
                             rhythm_status="rhythmic", period_underdetermined=False,
                             best_period_at_search_edge=False, best_period_hours=13.,
                             best_goodness_of_fit=.5, rhythmic=True, best_phase_hours=7.,
                             cosinor_amplitude=99., m10=999.)
    rhythms.loc[0, "rhythm_status"] = "unknown"
    rhythms.loc[1, "period_underdetermined"] = True
    rhythms.loc[2, "best_period_at_search_edge"] = True
    _, features, _, _, provenance = build_features({"cell_summary": summary, "rhythms": rhythms})
    period = next(c for c in features if "best_period_hours" in c)
    verdict = next(c for c in features if "|rhythmic|" in c)
    assert features.loc[:2, period].isna().all()
    assert features.loc[3:, period].eq(13).all()
    assert np.isnan(features.loc[0, verdict])
    assert not any(x in " ".join(features) for x in ("phase", "cosinor", "m10"))
    assert provenance[0]["period_estimation_method"] == "mesa"
    rhythms.loc[4, "period_estimation_method"] = "fft_nlls"
    with pytest.raises(ValueError, match="incompatible"):
        build_features({"cell_summary": summary, "rhythms": rhythms})


def test_preprocessing_does_not_learn_from_held_out_values():
    rng = np.random.default_rng(13)
    data = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    data["constant"] = 1
    data["duplicate"] = data.a * 2
    data.loc[:23, "missing"] = np.nan
    families = {c: "shape" for c in data}
    train = np.arange(24)
    options = ClusteringOptions()
    _, _, _, before = prepare_features(data, families, train, options)
    data.loc[24:, list("abcd")] = 1e12
    _, _, _, after = prepare_features(data, families, train, options)
    assert before == after
    assert set(before["excluded"]) == {"constant", "duplicate", "missing"}
    assert before["weights"] == [.5] * 4


def test_split_keeps_every_movie_of_an_animal_together():
    meta = measurements()
    train, validation = split_cells(meta, ClusteringOptions())
    assert len(validation)
    assert set(meta.iloc[train].subject).isdisjoint(set(meta.iloc[validation].subject))
    meta["subject"] = "one_animal"
    train, validation = split_cells(meta, ClusteringOptions())
    assert len(train) == len(meta)
    assert len(validation) == 0


def test_neural_replay_missing_cells_and_linear_reference():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    summary = measurements()
    cols = [c for c in summary if c.endswith("_median")]
    summary.loc[0, cols] = np.nan
    meta, features, families, _, _ = build_features({"cell_summary": summary})
    options = ClusteringOptions(epochs=35, patience=8, repeats=2,
                                min_cluster_size=5, min_samples=3, latent_dimensions=3)
    fitted = fit_cells(meta, features, families, options)
    assert fitted["assignments"].loc[0, "cluster"] == -2
    assert fitted["assignments"].loc[0, "split"] == "excluded"
    embedding, _ = replay_fingerprints(features, fitted["preprocessing"], fitted["checkpoint"])
    actual = fitted["embeddings"][["neural_1", "neural_2", "neural_3"]].to_numpy()
    np.testing.assert_allclose(embedding[1:], actual[1:], rtol=1e-6, atol=1e-6)
    report = fitted["report"]
    assert set(report["training_groups"]).isdisjoint(report["validation_groups"])
    assert len(report["neural_seed_sensitivity"]) == 1
    assert report["diagnostics"]["linear"]["clusters"] == 2
    assert report["biological_validation"] == "not established"


def test_saved_run_is_complete_replayable_and_immutable(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    run = tmp_path / "run"
    tables = run / "pooled" / "tables"
    tables.mkdir(parents=True)
    measurements().to_csv(tables / "cell_summary.csv", index=False)
    (tables.parent / "manifest.json").write_text(json.dumps({"tables": {}, "movies": ["movie_a", "movie_b"]}))
    out = tmp_path / "groups"
    options = ClusteringOptions(epochs=5, repeats=1, latent_dimensions=3, min_cluster_size=5)
    report = run_clustering(run, out, options)
    for name in ("assignments.csv", "profiles.csv", "composition.csv", "feature_audit.csv",
                 "preprocessing.json", "manifest.json", "neural_model.pt", "linear_model.npz"):
        assert (document(out / name) if name.endswith(".json") else out / name).is_file()
    checkpoint = torch.load(out / "neural_model.pt", weights_only=True)
    assert checkpoint["latent"] == 3
    assert len(report["inputs"]) == 2
    with pytest.raises(FileExistsError):
        run_clustering(run, out, options)


@pytest.mark.parametrize("settings", [{"epochs": 0}, {"seed": -1}, {"max_cell_missing": 1},
                                     {"noise_sd": float("nan")}, {"split_by": "condition"}])
def test_invalid_options_fail_before_training(settings):
    with pytest.raises(ValueError):
        ClusteringOptions(**settings)


def test_command_is_registered_without_training():
    from pymicroglia.registry import build_registry, pending
    assert "cluster" not in pending()
    assert callable(build_registry().resolve("cluster"))


def test_pool_availability_is_used_and_calibration_mismatch_refused(tmp_path):
    tables = tmp_path / "pooled" / "tables"
    tables.mkdir(parents=True)
    measurements().to_csv(tables / "cell_summary.csv", index=False)
    pool = {"tables": {"cell_summary": {"columns_missing": {"movie_a": ["signal_mean_median"]}}}}
    (tables.parent / "manifest.json").write_text(json.dumps(pool))
    loaded, provenance = load_tables(tmp_path)
    assert "signal_mean_median" not in loaded["cell_summary"]
    assert provenance["table_audit"][0]["reason"] == "column_not_measured_in_every_movie"
    movies = [{"stem": s, "provenance": {"scale": {"minutes_per_frame": 30, "microns_per_pixel": v}}}
              for s, v in [("movie_a", 1), ("movie_b", 2)]]
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": movies}))
    with pytest.raises(ValueError, match="microns_per_pixel"):
        load_tables(tmp_path)


def test_saved_rhythm_settings_are_required_and_must_match(tmp_path):
    tables = tmp_path / "pooled" / "tables"
    tables.mkdir(parents=True)
    measurements().to_csv(tables / "cell_summary.csv", index=False)
    pd.DataFrame({"stem": ["movie_a", "movie_b"]}).to_csv(tables / "rhythms.csv", index=False)
    (tables.parent / "manifest.json").write_text(json.dumps({"tables": {}}))
    loaded, provenance = load_tables(tmp_path)
    assert "rhythms" not in loaded
    assert provenance["table_audit"][0]["reason"] == "complete_saved_rhythm_settings_unavailable"
    movies = [{"stem": s, "modules": [{"module": "rhythms", "status": "done", "parameters": p}]}
              for s, p in [("movie_a", {"period_min_hours": 2}), ("movie_b", {"period_min_hours": 4})]]
    (tmp_path / "manifest.json").write_text(json.dumps({"movies": movies}))
    with pytest.raises(ValueError, match="incompatible saved rhythm settings"):
        load_tables(tmp_path)


def test_auxiliary_metadata_conflicts_are_errors():
    summary = measurements()
    frame = summary[["stem", "identity", "subject", "condition"]].assign(frame_index=0, area_px=10.)
    frame.loc[0, "subject"] = "wrong_animal"
    with pytest.raises(ValueError, match="inconsistent subject"):
        build_features({"cell_summary": summary, "cell_frame": frame})


def test_frame_channel_measurements_are_summarised_per_channel():
    summary = measurements()
    keys = summary[["stem", "identity"]]
    channels = pd.concat([keys.assign(frame_index=f, channel=c, channel_punctateness=v + f)
                          for c, v in [("red", 5.), ("green", 15.)] for f in range(3)])
    _, features, _, _, _ = build_features({"cell_summary": summary, "channels": channels})
    for c, expected in [("red", 6.), ("green", 16.)]:
        column = next(n for n in features if f'"{c}"' in n and n.endswith("|median"))
        assert features[column].eq(expected).all()
