"""Selectable methods retain scientific meaning and frozen training boundaries."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from test_states import frames, fast_options
from pymicroglia.states.features import frame_features
from pymicroglia.states.mixture import fit_states
from pymicroglia.states.dynamics import describe_dynamics
from pymicroglia.states.methods import replay_alternative


def inputs():
    return frame_features({"cell_frame": frames(animals=3, cells=2, count=36)})


def test_explicit_gaussian_default_is_identical():
    meta, snapshot, _, families, _ = inputs()
    options = fast_options(rhythm_enabled=False)
    default = fit_states(meta, snapshot, families, options)
    named = fit_states(meta, snapshot, families, replace(options, state_method="gaussian_mixture"))
    pd.testing.assert_frame_equal(default["assignments"], named["assignments"])
    assert named["report"]["score_kind"] == "posterior_probability"


def test_hard_indicators_produce_observed_occupancy_without_fake_probabilities():
    meta, snapshot, change, _, _ = frame_features({"cell_frame": frames(animals=1, cells=1, count=24)})
    meta["state"] = np.repeat([0, 1], 12)
    meta["state_indicator_0"] = (meta.state == 0).astype(float)
    meta["state_indicator_1"] = (meta.state == 1).astype(float)
    cells, transitions, _, dynamic = describe_dynamics(meta, snapshot, change)
    assert cells.occupancy_kind.iloc[0] == "hard_assignment"
    assert np.isnan(cells.probability_observed_hours.iloc[0])
    assert cells.state_observed_hours.iloc[0] == 11.5
    assert cells.switches.iloc[0] == 1
    assert dynamic.filter(like="occupancy|").sum(axis=1).iloc[0] == pytest.approx(1)


@pytest.mark.parametrize("method", ["kmeans", "agglomerative", "hdbscan"])
def test_distance_density_replay_and_training_isolation(method):
    meta, snapshot, _, families, _ = inputs()
    options = replace(fast_options(rhythm_enabled=False), state_method=method,
                      candidate_states=(2,), density_min_cluster_size=8, density_min_samples=3)
    fitted = fit_states(meta, snapshot, families, options)
    assert not any(c.startswith("state_probability_") for c in fitted["assignments"])
    replay = replay_alternative(meta, snapshot, fitted["preprocessing"], fitted["model"], fitted["report"])
    pd.testing.assert_frame_equal(replay.drop(columns="split"), fitted["assignments"].drop(columns="split"))
    changed = snapshot.copy()
    changed.loc[fitted["assignments"].split.eq("test")] += 1000
    second = fit_states(meta, changed, families, options)
    np.testing.assert_array_equal(fitted["model"]["centres"], second["model"]["centres"])
    if method == "agglomerative":
        assert len(fitted["model"]["merge_children"]) == len(fitted["model"]["anchor_values"]) - 1
        assert set(replay.assignment_origin) == {"native_training_partition", "nearest_centroid_extension"}
    elif method == "hdbscan":
        assert "membership_strength" in replay
        assert replay.membership_strength.between(0, 1).all()
        assert fitted["report"]["states"] >= 2


def test_density_all_noise_and_missing_frames_are_supported():
    meta, snapshot, change, families, _ = frame_features({"cell_frame": frames(animals=1, cells=1, count=36)})
    snapshot.iloc[0] = np.nan
    options = replace(fast_options(rhythm_enabled=False), state_method="hdbscan",
                      density_min_cluster_size=30, density_min_samples=3)
    fitted = fit_states(meta, snapshot, families, options)
    assert fitted["report"]["states"] == 0
    assert fitted["assignments"].state.iloc[0] == -2
    assert fitted["assignments"].state.iloc[1:].eq(-1).all()
    cells, transitions, _, _ = describe_dynamics(fitted["assignments"], snapshot, change)
    assert transitions.empty
    assert cells.state_observed_hours.eq(0).all()


def test_density_replay_never_refits(monkeypatch):
    import hdbscan
    meta, snapshot, _, families, _ = inputs()
    options = replace(fast_options(), state_method="hdbscan", density_min_cluster_size=8, density_min_samples=3)
    fitted = fit_states(meta, snapshot, families, options)
    def forbidden(*args, **kwargs):
        raise AssertionError("A frozen dictionary must not fit again")
    monkeypatch.setattr(hdbscan.HDBSCAN, "fit", forbidden)
    replay = replay_alternative(meta, snapshot, fitted["preprocessing"], fitted["model"], fitted["report"])
    np.testing.assert_array_equal(replay.state, fitted["assignments"].state)


@pytest.mark.parametrize("linkage", ["ward", "average", "complete", "single"])
def test_hierarchical_linkages_are_available(linkage):
    meta, snapshot, _, families, _ = inputs()
    result = fit_states(meta, snapshot, families, replace(fast_options(), state_method="agglomerative", agglomerative_linkage=linkage))
    assert result["report"]["linkage"] == linkage


def test_hidden_markov_replay_isolation_and_missing_boundaries():
    from pymicroglia.states.hmm import sequences
    meta, snapshot, _, families, _ = inputs()
    options = replace(fast_options(rhythm_enabled=False), state_method="hidden_markov", candidate_states=(2,), max_iterations=500)
    fitted = fit_states(meta, snapshot, families, options)
    replay = replay_alternative(meta, snapshot, fitted["preprocessing"], fitted["model"], fitted["report"])
    pd.testing.assert_frame_equal(replay.drop(columns="split"), fitted["assignments"].drop(columns="split"))
    np.testing.assert_allclose(fitted["model"]["transmat"].sum(axis=1), 1)
    changed = snapshot.copy()
    changed.loc[fitted["assignments"].split.eq("test")] += 1000
    second = fit_states(meta, changed, families, options)
    np.testing.assert_array_equal(second["model"]["transmat"], fitted["model"]["transmat"])
    good = np.ones(len(meta), bool)
    good[10] = False
    parts, cadence = sequences(meta, good)
    assert cadence == .5
    assert not any(9 in p and 11 in p for p in parts)
    assert all(len(meta.iloc[p][["stem", "identity"]].drop_duplicates()) == 1 for p in parts)
    irregular = meta.copy()
    irregular.loc[10, "hours"] += .1
    with pytest.raises(ValueError, match="cadence"):
        sequences(irregular, np.ones(len(meta), bool))


def test_hidden_markov_uses_order_and_restarts_at_gaps():
    from pymicroglia.states.hmm import infer_hidden_markov
    meta, _, _, _, _ = frame_features({"cell_frame": frames(animals=1, cells=1, count=24)})
    arrays = {"startprob": np.array([.5, .5]), "transmat": np.array([[.99, .01], [.01, .99]]),
              "means": np.array([[-1.], [1.]]), "covariances": np.array([[1.], [1.]]), "emission_weights": np.array([.5, .5])}
    report = {"states": 2, "covariance_type": "diag", "max_frame_missing": .3, "density_threshold": None,
              "max_gap_hours": None, "cadence_hours": .5, "state_probability_min": .6}
    values = np.repeat([-1., 1.], 12).reshape(-1, 1)
    ordered = infer_hidden_markov(meta, values, np.ones(24), arrays, report)
    permutation = np.ravel(np.column_stack([np.arange(12), np.arange(12, 24)]))
    mixed = infer_hidden_markov(meta, values[permutation], np.ones(24), arrays, report)
    assert not np.allclose(ordered.state_probability_0.to_numpy()[permutation], mixed.state_probability_0)
    gapped = meta.copy()
    gapped.loc[12:, "frame_index"] += 1
    gapped.loc[12:, "hours"] += .5
    first = infer_hidden_markov(gapped, values, np.ones(24), arrays, report)
    altered = values.copy(); altered[:12] *= -1
    second = infer_hidden_markov(gapped, altered, np.ones(24), arrays, report)
    np.testing.assert_allclose(first.state_probability_0.iloc[12:], second.state_probability_0.iloc[12:])


@pytest.mark.parametrize("covariance", ["diag", "full", "tied", "spherical"])
def test_hidden_markov_covariances_replay(covariance):
    meta, snapshot, _, families, _ = inputs()
    fitted = fit_states(meta, snapshot, families, replace(fast_options(), state_method="hidden_markov",
                       candidate_states=(2,), covariance_type=covariance, max_iterations=500))
    replay = replay_alternative(meta, snapshot, fitted["preprocessing"], fitted["model"], fitted["report"])
    np.testing.assert_allclose(replay.filter(like="state_probability_"), fitted["assignments"].filter(like="state_probability_"))


def test_trajectory_distance_respects_order_and_physical_warp_limit():
    from pymicroglia.states.trajectory import bounded_distances
    up = np.arange(12, dtype=float)
    series = np.stack([up, up, up[::-1]])[:, :, None]
    distance, radius = bounded_distances(series, .5, .75)
    assert radius == 1
    assert distance[0, 1] == 0
    assert distance[0, 2] > 0
    same_time, radius = bounded_distances(series, .5, 0)
    assert radius == 0
    assert same_time[0, 2] == pytest.approx(np.sqrt(np.mean((up - up[::-1]) ** 2)))


def test_trajectory_common_window_excludes_gaps_and_keeps_original_time():
    from pymicroglia.states.trajectory import group_trajectories
    table = frames(animals=1, cells=8, count=36)
    table = table.loc[~((table.identity == 7) & (table.frame_index == 18))]
    meta, snapshot, change, _, _ = frame_features({"cell_frame": table})
    options = replace(fast_options(), dynamic_method="trajectory_dtw", trajectory_groups=2,
                      trajectory_window_start_hours=0, trajectory_window_end_hours=17.5)
    result = group_trajectories(meta, snapshot, change, options)
    assert result["report"]["status"] == "fitted"
    assert result["report"]["cells_compared"] == 7
    assert result["assignments"].set_index("identity").loc[7, "dynamic_group"] == -2
    assert result["report"]["max_warp_frames"] == 2
    assert result["trajectories"].hours.min() == 0
    assert result["trajectories"].hours.max() == 17.5
    assert result["trajectories"].groupby("identity").size().eq(36).all()
