"""Small neural fingerprint model and a linear reference for cell clustering.

No rhythm calculations live here. Missing measurements contribute no neural
reconstruction loss; all preprocessing is learned from training cells only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


from .options import ClusteringOptions


def split_cells(metadata: pd.DataFrame, options: ClusteringOptions):
    """Hold out entire declared subjects (or explicitly requested movies)."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = metadata[options.split_by]
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        raise ValueError(f"Every cell needs {options.split_by} metadata for the split")
    if groups.nunique() < 2 or options.validation_fraction == 0:
        return np.arange(len(metadata)), np.array([], dtype=int)
    return next(GroupShuffleSplit(n_splits=1, test_size=options.validation_fraction,
                                  random_state=options.seed).split(metadata, groups=groups))


def prepare_features(features: pd.DataFrame, families: dict[str, str],
                     train: np.ndarray, options: ClusteringOptions):
    """Filter, impute and scale using training values; balance measurement families."""
    finite = features.replace([np.inf, -np.inf], np.nan)
    fitting = finite.iloc[train]
    reasons = {}
    usable = []
    for column in finite:
        values = fitting[column]
        if values.isna().mean() > options.max_feature_missing:
            reasons[column] = "too_many_missing_training_values"
        elif values.nunique() < 2:
            reasons[column] = "constant_in_training"
        else:
            usable.append(column)
    if len(usable) < 2:
        raise ValueError("Fewer than two variable, sufficiently observed training measurements")
    # Remove redundant versions of a measurement before a family can dominate.
    corr = fitting[usable].corr(method="spearman", min_periods=3).abs()
    kept = []
    for column in usable:
        redundant = next((c for c in kept if corr.loc[c, column] >= options.correlation_cutoff), None)
        if redundant is None:
            kept.append(column)
        else:
            reasons[column] = f"redundant_with:{redundant}"
    if len(kept) < 2:
        raise ValueError("Fewer than two nonredundant measurements remain")
    fitting = fitting[kept]
    median = fitting.median()
    scale = fitting.quantile(0.75) - fitting.quantile(0.25)
    scale = scale.where(scale > 1e-12, fitting.std(ddof=0)).clip(lower=1e-12)
    counts = pd.Series([families[c] for c in kept]).value_counts()
    weights = np.array([1 / np.sqrt(counts[families[c]]) for c in kept])
    state = {"columns": kept, "median": median.tolist(), "scale": scale.tolist(),
             "weights": weights.tolist(), "families": [families[c] for c in kept],
             "excluded": reasons}
    values, observed = transform_features(finite, state)
    eligible = observed.mean(axis=1) >= 1 - options.max_cell_missing
    return values, observed, eligible, state


def transform_features(features: pd.DataFrame, state: dict):
    """Reuse fitted preprocessing without learning from the incoming cells."""
    raw = features.reindex(columns=state["columns"]).to_numpy(dtype=float)
    observed = np.isfinite(raw)
    median = np.asarray(state["median"])
    values = (np.where(observed, raw, median) - median) / np.asarray(state["scale"])
    values = np.clip(values, -10, 10) * np.asarray(state["weights"])
    return values.astype(np.float32), observed


def _network(torch, inputs, hidden, latent):
    return torch.nn.Sequential(
        torch.nn.Linear(inputs, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, max(latent * 2, hidden // 2)), torch.nn.ReLU(),
        torch.nn.Linear(max(latent * 2, hidden // 2), latent),
        torch.nn.Linear(latent, max(latent * 2, hidden // 2)), torch.nn.ReLU(),
        torch.nn.Linear(max(latent * 2, hidden // 2), hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, inputs),
    )


def learn_fingerprints(values, observed, train, validation, options, seed):
    """Fit a denoising autoencoder on CPU; return embeddings and replayable weights."""
    import torch

    latent = min(options.latent_dimensions, values.shape[1] - 1, len(train) - 1)
    if latent < 1:
        raise ValueError("Insufficient training cells or measurements")
    # Keep the caller's random state and thread configuration intact.
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = _network(torch, values.shape[1], options.hidden_width, latent)
            x = torch.from_numpy(values)
            mask = torch.from_numpy(observed.astype(np.float32))
            optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate,
                                           weight_decay=options.weight_decay)

            def loss(prediction, indices):
                squared = (prediction - x[indices]).square() * mask[indices]
                return (squared.sum(1) / mask[indices].sum(1).clamp(min=1)).mean()

            best = float("inf")
            best_state = None
            stale = 0
            history = []
            for epoch in range(options.epochs):
                model.train()
                noisy = x[train] + torch.randn_like(x[train]) * options.noise_sd * mask[train]
                optimizer.zero_grad()
                training_loss = loss(model(noisy), train)
                training_loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    score = float(loss(model(x[validation]), validation)) if len(validation) else float(loss(model(x[train]), train))
                if not np.isfinite(score):
                    raise ValueError("Neural training produced a nonfinite loss")
                history.append({"epoch": epoch + 1, "training_loss": float(training_loss.detach()),
                                "validation_loss": score if len(validation) else None})
                if score < best - 1e-7:
                    best, stale = score, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                if len(validation) and stale >= options.patience:
                    break
            model.load_state_dict(best_state)
            with torch.no_grad():
                embedding = model[:5](x).numpy().copy()
                error = (((model(x) - x).square() * mask).sum(1)
                         / mask.sum(1).clamp(min=1)).numpy().copy()
            checkpoint = {"state_dict": best_state, "inputs": values.shape[1],
                          "hidden": options.hidden_width, "latent": latent,
                          "seed": seed, "options": asdict(options)}
    finally:
        torch.set_num_threads(threads)
    return embedding, error, checkpoint, history


def replay_fingerprints(features: pd.DataFrame, preprocessing: dict, checkpoint: dict):
    """Encode new measurements with the saved model; this does not assign clusters."""
    import torch

    values, observed = transform_features(features, preprocessing)
    with torch.random.fork_rng(devices=[]):
        model = _network(torch, checkpoint["inputs"], checkpoint["hidden"], checkpoint["latent"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            embedding = model[:5](torch.from_numpy(values)).numpy().copy()
    return embedding, observed.mean(axis=1)
