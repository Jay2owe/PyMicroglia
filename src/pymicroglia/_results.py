"""Read saved analyses across the old and current layout; file records once."""
from pathlib import Path
import json


def document(path):
    from auto_organotypic.layout import document as locate
    path = Path(path)
    return locate(path.parent, path.name)


def read_document(path, **kwargs):
    from auto_organotypic.workbench_records import unfold
    return unfold(json.loads(document(path).read_text(encoding="utf-8"), **kwargs))


def write_document(path, value):
    from auto_organotypic import io
    # Reject nonfinite provenance before the atomic writer touches anything.
    json.dumps(value, allow_nan=False)
    return io.write_json(path, value, workings=True)


def output_files(folder):
    """Visible deliverables and hidden documents, excluding the folder ledger."""
    from auto_organotypic import layout
    folder=Path(folder)
    return [p for root in (folder,layout.workings_path(folder)) if root.is_dir()
            for p in root.iterdir() if p.is_file() and p.name!='artefacts.json' and not p.name.startswith('.')]


def pooled_paths(run):
    root = Path(run).resolve()
    pooled = root / "pooled" if (root / "pooled").is_dir() else root
    parent = pooled.parent if pooled.name == "pooled" else root
    root_manifest = document(parent / "manifest.json")
    pool_manifest = document(pooled / "manifest.json")
    if not pool_manifest.is_file():
        pool_manifest = root_manifest
    tables = pooled / "tables" if (pooled / "tables").is_dir() else pooled
    return pool_manifest, root_manifest, tables


def register_outputs(folder, *, stage, inputs, settings):
    """Index existing tables and models without changing their bytes."""
    from . import store, __version__
    folder = Path(folder)
    paths = [Path(row["path"]) for row in inputs]
    if not paths:
        raise ValueError("Result records need at least one source file")
    source = store.collection(paths, path=str(folder))
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != "artefacts.json":
            store.claim(stage, source, {**settings, "output": path.name},
                        path=path, display_only=False, method_version=__version__)


def copy_artifact(source, target):
    """Copy immutable bytes together with their shared-ledger type contract."""
    import shutil
    from auto_organotypic import store
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    entry=store.ledger.entry_for(source)
    if entry is not None:store.ledger.record(target,entry)
    return target


def report_name(path, root):
    """Stable artifact names are independent of the machine-document directory."""
    from auto_organotypic.layout import WORKINGS
    return Path(*(part for part in Path(path).relative_to(root).parts if part!=WORKINGS)).as_posix()


def workings_link(name):
    from auto_organotypic.layout import WORKINGS
    return (Path(WORKINGS)/name).as_posix()



SOURCE_IDENTITY_KEY = "measurement_source_identity"


def measurement_identity(path):
    """Retain the original manifest byte hash when adding display-only plans."""
    import hashlib
    from .pipelines._contracts import content_id
    path = Path(path)
    located = document(path)
    manifest = read_document(located)
    base = {key: value for key, value in manifest.items()
            if key not in {"figure_plans", SOURCE_IDENTITY_KEY}}
    recorded = manifest.get(SOURCE_IDENTITY_KEY)
    if recorded:
        if recorded["content_sha256"] != content_id(base):
            raise ValueError("Measurement manifest changed after its source identity was recorded")
        return recorded["manifest_sha256"]
    if "figure_plans" not in manifest:
        return hashlib.sha256(located.read_bytes()).hexdigest()
    # Early port candidates used a content identity before this preservation
    # field existed. Their saved reports remain readable; fresh runs use bytes.
    return content_id(base)


def figure_plans(run, *, manifest=None):
    """Read presentation plans without changing the source measurement record."""
    run = Path(run)
    if manifest is None:
        try:
            manifest = read_document(run / "manifest.json")
        except FileNotFoundError:
            manifest = {}
    plans = dict(manifest.get("figure_plans", {}))
    try:
        separate = read_document(run / "figure-plans.json").get("figure_plans", {})
    except FileNotFoundError:
        separate = {}
    for name, item in separate.items():
        if name in plans and plans[name] != item:
            raise ValueError(f"Conflicting saved figure plan: {name}")
        plans[name] = item
    return plans
