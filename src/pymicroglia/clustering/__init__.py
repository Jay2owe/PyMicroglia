"""Whole-cell fingerprints and descriptive cohort clusters."""


def cluster(run, *, output_dir=None, clustering_options=None, if_exists="version",
            run_label=None, claim=""):
    from pathlib import Path
    from ..pipelines import run_folder
    from ..registry import require_claim
    from .._results import read_document
    from .options import ClusteringOptions
    require_claim("cluster", claim)
    try:
        import sklearn
        import torch
    except ImportError as error:
        raise ImportError('Cell fingerprints require pip install "PyMicroglia[states,mask]"') from error
    from .cohort import run_clustering
    options = ClusteringOptions(**dict(clustering_options or {}))
    root = Path(output_dir) if output_dir else Path(run) / "clusters"
    target = run_folder(root.parent, root.name, run_label or "clusters", if_exists)
    if target.reuse:
        return read_document(Path(target.path) / "manifest.json")
    return run_clustering(run, target.path, options)


def __getattr__(name):
    if name == "ClusteringOptions":
        from .options import ClusteringOptions
        return ClusteringOptions
    raise AttributeError(name)
