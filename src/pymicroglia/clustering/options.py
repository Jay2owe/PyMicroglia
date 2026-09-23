"""Validated learning settings; importing these never loads learning libraries."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

@dataclass(frozen=True)
class ClusteringOptions:
    latent_dimensions: int = 8
    hidden_width: int = 32
    epochs: int = 300
    patience: int = 30
    learning_rate: float = 0.001
    noise_sd: float = 0.1
    weight_decay: float = 0.001
    max_feature_missing: float = 0.3
    max_cell_missing: float = 0.3
    correlation_cutoff: float = 0.98
    min_cluster_size: int = 8
    min_samples: int = 5
    validation_fraction: float = 0.2
    split_by: str = "subject"
    seed: int = 42
    repeats: int = 3

    def __post_init__(self):
        for name in ("latent_dimensions", "hidden_width", "epochs", "patience",
                     "min_cluster_size", "min_samples", "repeats"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("max_feature_missing", "max_cell_missing", "validation_fraction"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 < self.correlation_cutoff <= 1:
            raise ValueError("correlation_cutoff must be in (0, 1]")
        for name in ("learning_rate", "noise_sd", "weight_decay"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if self.split_by not in ("subject", "stem"):
            raise ValueError("split_by must be subject or stem")
