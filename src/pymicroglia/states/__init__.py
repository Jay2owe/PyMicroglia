"""Learn and replay a shared vocabulary of measured cell-frame states."""


def states(run, *, output_dir=None, state_method="gmm", states=None, features=None,
           holdout="group", seed=0, replay=None, rhythms=True, trajectory=False,
           state_options=None, if_exists="version", run_label=None, claim=""):
    from pathlib import Path
    from ..pipelines import run_folder
    from ..registry import require_claim
    from .._results import read_document
    from .options import StateOptions
    require_claim("states", claim)
    try:
        import sklearn
    except ImportError as error:
        raise ImportError('State learning requires pip install "PyMicroglia[states]"') from error
    from .engine import run_states
    values = dict(state_options or {})
    values.setdefault("state_method", {"gmm": "gaussian_mixture", "hmm": "hidden_markov"}.get(state_method, state_method))
    values.setdefault("seed", seed)
    values.setdefault("rhythm_enabled", rhythms)
    if states is not None:
        values["candidate_states"] = (states,) if isinstance(states, int) else tuple(states)
    if features is not None:
        values["snapshot_features"] = tuple(features)
    if holdout not in {"group", "subject", "stem", "cells"}:
        raise ValueError("holdout must be group, subject, stem or cells")
    if holdout == "cells":
        values["selection_scope"] = "cells"
    elif holdout in {"subject", "stem"}:
        values["split_by"] = holdout
    if trajectory:
        values.update(group_dynamic_cells=True, dynamic_method="trajectory_dtw")
    options = StateOptions(**values)
    root = Path(output_dir) if output_dir else Path(run) / "states"
    target = run_folder(root.parent, root.name, run_label or "states", if_exists)
    if target.reuse:
        return read_document(Path(target.path) / "manifest.json")
    return run_states(run, target.path, options, model=replay)


def __getattr__(name):
    if name == "StateOptions":
        from .options import StateOptions
        return StateOptions
    if name in {"run_states", "apply_dictionary"}:
        from . import engine
        return getattr(engine, name)
    raise AttributeError(name)
