"""Small reproducible pixels -> real measurements fixture for the live UI."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import tifffile


def create_fixture(folder: Path):
    folder = folder.resolve()
    recipe = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt = folder / "fixture.json"
    if receipt.is_file():
        old = json.loads(receipt.read_text())
        if old.get("recipe_sha256") == recipe and Path(old["run"]).is_dir():
            return old
        raise ValueError("This fixture was created by a different recipe; choose a new output folder.")
    if shutil.disk_usage(folder.parent if folder.parent.exists() else Path.cwd()).free < 512 * 1024**2:
        raise ValueError("At least 512 MiB of free space is needed for this fixture.")
    folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:160, :240]
    rng = np.random.default_rng(8426)
    movies = []
    fingerprints = {}
    identities = [19, 26, 31, 34, 40, 45, 51, 55, 61, 67, 73, 80]
    for recording in range(2):
        raw = np.zeros((164, 160, 240), np.uint16)
        labels = np.zeros((160, 160, 240), np.uint16)
        for source_frame in range(164):
            f = source_frame - 4
            plane = 100 + 30 * np.sin(xx / 65) + 20 * np.cos(yy / 35) + rng.normal(0, 10, xx.shape)
            for idx, identity in enumerate(identities):
                cx = 36 + (idx % 4) * 53 + 4 * np.sin(f / 21 + idx)
                cy = 30 + (idx // 4) * 48 + 3 * np.cos(f / 16 + idx)
                angle = np.arctan2(yy - cy, xx - cx)
                radius = np.hypot(yy - cy, xx - cx)
                boundary = 8 + 2 * np.sin(f / 17 + idx) + 4 * np.cos(3 * angle + f / 30 + idx) + 2 * np.cos(5 * angle - idx)
                mask = radius <= boundary
                phase = idx / 12 * np.pi * 2 + recording * .3
                intensity = 800 + 450 * np.cos(2 * np.pi * f / 48 - phase)
                plane += mask * intensity + 300 * np.exp(-((xx-cx)**2+(yy-cy)**2)/12)
                if f >= 0 and not (identity == 19 and 73 <= f <= 76) and not (identity == 26 and f in (15, 16)):
                    labels[f, mask] = identity
            raw[source_frame] = np.clip(plane, 0, 65535).astype(np.uint16)
        name = f"synthetic_{recording + 1}"
        for role, data in (("raw", raw), ("labels", labels), ("base_labels", np.concatenate([labels[:1], labels[:-1]]))):
            path = folder / f"{name}_{role}.tif"
            tifffile.imwrite(path, data, photometric="minisblack", metadata={"axes": "TYX"})
            fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        movies.append({"stem": name, "raw": str(folder / f"{name}_raw.tif"), "labels": str(folder / f"{name}_labels.tif"),
                       "source_frame_offset": 4, "subject": f"synthetic subject {recording + 1}"})
    from pymicroglia.measure import measure
    measure(movies, output_dir=folder / "measurements", run_label="saved-ui-fixture", frame_interval_min=30,
            enabled_modules=["morphology", "intensity", "motility", "contacts", "lifecycle", "rhythms"],
            module_options={"rhythms": {"metrics": ["integrated_density", "area_px"], "period_methods": ["lomb"],
                                        "period_search_hours": [20, 28], "null_surrogates_per_cell": 0}},
            windows=[{"name": "early", "from_frame": 0, "to_frame": 80}, {"name": "late", "from_frame": 80, "to_frame": 160}],
            claim="Verify linked exploration with reproducible synthetic pixels, fixed identities and real saved measurements.")
    run = folder / "measurements/saved-ui-fixture"
    from pymicroglia.measure.run import read_manifest, write_manifest
    saved = read_manifest(run)
    for movie in saved["movies"]:
        movie["display"] = {"base_labels": str(folder / (movie["stem"] + "_base_labels.tif")), "contrast": [60, 1550]}
    write_manifest(run, saved)
    result = {"run": str(run), "synthetic": True, "recipe_sha256": recipe, "input_sha256": fingerprints,
              "description": "Two synthetic recordings, 12 fixed identities, 160 analysed frames, raw source offset 4. Measurement and rhythm tables were computed by the installed analysis APIs."}
    receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_fixture(args.folder), indent=2))
