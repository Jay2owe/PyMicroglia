"""Validated learning settings; importing these never loads learning libraries."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

@dataclass(frozen=True)
class StateOptions:
    state_method: str = "gaussian_mixture"
    distance_margin_min: float = 0.0
    agglomerative_linkage: str = "ward"
    hierarchical_max_samples: int = 5000
    density_min_cluster_size: int = 50
    density_min_samples: int = 10
    density_selection_method: str = "eom"
    hmm_tolerance: float = 0.001
    representation: str = "measurements"
    candidate_states: tuple[int, ...] | None = None
    auto_initial_states: int = 6
    auto_max_states: int = 24
    selection_scope: str = "auto"
    covariance_type: str = "diag"
    regularization: float = 0.001
    initializations: int = 5
    max_iterations: int = 300
    min_cell_frames: int = 12
    training_samples: int = 20000
    max_feature_missing: float = 0.3
    max_frame_missing: float = 0.3
    correlation_cutoff: float = 0.98
    state_probability_min: float = 0.6
    outlier_fraction: float = 0.01
    split_by: str = "subject"
    selection_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 42
    stability_repeats: int = 3
    max_gap_hours: float | None = None
    snapshot_features: tuple[str, ...] | None = None
    neural: dict = field(default_factory=dict)
    rhythm_enabled: bool = True
    rhythm_workers: int = 1
    rhythm_metrics: tuple[str, ...] | None = None
    rhythms: dict = field(default_factory=dict)
    persistence_surrogates: int = 20
    group_dynamic_cells: bool = True
    dynamic_min_cluster_size: int = 5
    dynamic_min_samples: int = 3
    dynamic_method: str = "descriptor_hdbscan"
    trajectory_groups: int | None = None
    trajectory_min_cells: int = 5
    trajectory_max_warp_hours: float = 1.0
    trajectory_window_start_hours: float | None = None
    trajectory_window_end_hours: float | None = None

    def __post_init__(self):
        if self.state_method not in {"gaussian_mixture", "kmeans", "agglomerative", "hdbscan", "hidden_markov"}:
            raise ValueError("Unknown state_method")
        if self.dynamic_method not in {"descriptor_hdbscan", "trajectory_dtw"}:
            raise ValueError("Unknown dynamic_method")
        for name in ("trajectory_groups", "trajectory_min_cells", "auto_initial_states", "auto_max_states"):
            value = getattr(self, name)
            if name == "trajectory_groups" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.trajectory_max_warp_hours) or self.trajectory_max_warp_hours < 0:
            raise ValueError("trajectory_max_warp_hours must be finite and nonnegative")
        start, end = self.trajectory_window_start_hours, self.trajectory_window_end_hours
        if (start is None) != (end is None) or (start is not None and
                (not np.isfinite(start) or not np.isfinite(end) or end <= start)):
            raise ValueError("Trajectory window requires finite start and end hours with end > start")
        if self.agglomerative_linkage not in {"ward", "average", "complete", "single"}:
            raise ValueError("Unknown agglomerative_linkage")
        if self.density_selection_method not in {"eom", "leaf"}:
            raise ValueError("density_selection_method must be eom or leaf")
        if not 0 <= self.distance_margin_min <= 1:
            raise ValueError("distance_margin_min must be in [0, 1]")
        if not np.isfinite(self.hmm_tolerance) or self.hmm_tolerance <= 0:
            raise ValueError("hmm_tolerance must be positive and finite")
        for name in ("hierarchical_max_samples", "density_min_cluster_size", "density_min_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.representation not in {"measurements", "neural"}:
            raise ValueError("representation must be measurements or neural")
        if self.auto_initial_states > self.auto_max_states:
            raise ValueError("auto_initial_states cannot exceed auto_max_states")
        if self.selection_scope not in {"auto", "independent_groups", "cells"}:
            raise ValueError("selection_scope must be auto, independent_groups or cells")
        if self.candidate_states is not None and (not self.candidate_states or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in self.candidate_states)):
            raise ValueError("candidate_states must contain positive integers")
        if self.covariance_type not in {"diag", "full", "tied", "spherical"}:
            raise ValueError("Unknown Gaussian mixture covariance_type")
        for name in ("initializations", "max_iterations", "min_cell_frames", "training_samples",
                     "stability_repeats", "dynamic_min_cluster_size", "dynamic_min_samples", "rhythm_workers"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dynamic_min_cluster_size < 2:
            raise ValueError("dynamic_min_cluster_size must be at least two")
        if not isinstance(self.persistence_surrogates, int) or isinstance(self.persistence_surrogates, bool) or self.persistence_surrogates < 0:
            raise ValueError("persistence_surrogates must be a nonnegative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("max_feature_missing", "max_frame_missing", "outlier_fraction", "selection_fraction", "test_fraction"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.selection_fraction + self.test_fraction >= 1:
            raise ValueError("Selection and test fractions must leave training groups")
        if not 0 <= self.state_probability_min <= 1 or not 0 < self.correlation_cutoff <= 1:
            raise ValueError("Invalid probability or correlation threshold")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be positive and finite")
        if self.max_gap_hours is not None and (not np.isfinite(self.max_gap_hours) or self.max_gap_hours <= 0):
            raise ValueError("max_gap_hours must be positive and finite")
        if self.split_by not in {"subject", "stem"}:
            raise ValueError("split_by must be subject or stem")
        for name in ("rhythm_enabled", "group_dynamic_cells"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        for name in ("neural", "rhythms"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be an object")
        for name in ("snapshot_features", "rhythm_metrics"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (list, tuple)) or not value or any(not isinstance(v, str) for v in value)):
                raise ValueError(f"{name} must be a nonempty list or null")
