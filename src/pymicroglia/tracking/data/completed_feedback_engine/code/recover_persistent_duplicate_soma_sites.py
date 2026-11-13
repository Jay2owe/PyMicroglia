"""Separate persistent soma sites that inherited one early identity.

Discovery uses soma positions and accepted labels, never feedback or movie IDs.
The original foreground and previously accepted changes remain protected.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

import recover_terminal_emergent_name_theft as terminal

HERE = Path(__file__).resolve()
TUNING = HERE.parents[1]
ROOT = next((p for p in HERE.parents if (p / "WORKFLOW_CONTRACT.md").exists()), HERE.parent)
OUT = TUNING / "diagnostics/persistent_duplicate_soma_sites_r119"
STRUCTURE = np.ones((3, 3), bool)


def _site(seed, by_frame: dict, n_frames: int) -> pd.DataFrame:
    """Follow one near-stationary soma by reciprocal spatial proximity."""
    rows = [seed]
    start, sx, sy, sr = (int(seed.frame), float(seed.x), float(seed.y),
                         max(float(seed.radius_px), 1.0))
    last = seed
    for t in range(start + 1, n_frames):
        group = by_frame.get(t)
        if group is None or group.empty:
            continue
        distance = np.hypot(group.x.to_numpy(float)-float(last.x),
                            group.y.to_numpy(float)-float(last.y))
        seed_distance = np.hypot(group.x.to_numpy(float)-sx,
                                 group.y.to_numpy(float)-sy)
        gap = t-int(last.frame)
        limit = np.maximum(7.0, 1.6*(float(last.radius_px)
                                      +group.radius_px.to_numpy(float)))
        valid = ((distance <= limit*min(gap, 2))
                 & (seed_distance <= max(11.0, 4.0*sr)))
        choices = np.flatnonzero(valid)
        if not len(choices):
            if gap > 4:
                break
            continue
        cost = distance[choices]/np.maximum(
            float(last.radius_px)+group.radius_px.to_numpy(float)[choices], 1.)
        best_local = int(np.argmin(cost))
        if len(cost)>1 and np.partition(cost,1)[1]-cost[best_local] < .15:
            if gap > 4:
                break
            continue
        row = group.iloc[int(choices[best_local])]
        rows.append(row)
        last = row
    return pd.DataFrame(rows)


def _candidate(first: pd.DataFrame, second: pd.DataFrame,
               name: int, all_points: pd.DataFrame) -> dict | None:
    if len(first) < 60 or len(second) < 60:
        return None
    first = first.drop_duplicates("frame").sort_values("frame")
    second = second.drop_duplicates("frame").sort_values("frame")
    early = first[first.frame < int(second.frame.min())]
    if len(early) < 1 or not early.owner.eq(name).all():
        return None
    shared = first.merge(second, on="frame", suffixes=("_a", "_b"))
    if len(shared) < 50:
        return None
    separation = np.hypot(shared.x_a-shared.x_b, shared.y_a-shared.y_b)
    radii = shared.radius_px_a+shared.radius_px_b
    if float(np.percentile(separation/np.maximum(radii, 1.), 5)) < 3.0:
        return None
    simultaneous_a = shared[(shared.owner_a == name) & (shared.owner_b == name)]
    if len(simultaneous_a) < 5:
        return None
    later = first[first.frame > int(simultaneous_a.frame.max())]
    names = later.owner.astype(int)
    names = names[(names > 0) & (names != name)]
    if len(names) < 30:
        return None
    wrong = int(names.mode().iloc[0])
    if int((names == wrong).sum()) < 30:
        return None
    if len(first[first.owner == name]) < 8:
        return None
    switch = int(later[later.owner == wrong].frame.min())
    after = second[second.frame >= switch]
    if len(after) < 30 or int(after.owner.eq(name).sum()) < .8*len(after):
        return None
    first_wrong = first[(first.frame >= switch) & (first.owner > 0)
                        & (first.owner != name)]
    if len(first_wrong) < 30:
        return None
    # The later wrong name also belongs to another independently observed soma.
    other = all_points[(all_points.frame.isin(first_wrong.frame))
                       & all_points.owner.eq(wrong)].copy()
    other = other.merge(first_wrong[["frame", "x", "y"]], on="frame",
                        suffixes=("", "_target"))
    other = other[np.hypot(other.x-other.x_target,
                           other.y-other.y_target) > 12]
    if other.frame.nunique() < 3:
        return None
    return {
        "original_name": name, "temporary_name": wrong,
        "first_site_first_frame": int(first.frame.min())+1,
        "second_site_first_frame": int(second.frame.min())+1,
        "first_site_observed_frames": len(first),
        "second_site_observed_frames": len(second),
        "first_site_last_frame": int(first.frame.max())+1,
        "second_site_last_frame": int(second.frame.max())+1,
        "early_duplicate_frames": len(simultaneous_a),
        "later_first_site_wrong_frames": len(first_wrong),
        "later_second_site_original_frames": int(after.owner.eq(name).sum()),
        "other_temporary_name_frames": int(other.frame.nunique()),
        "separation_p05_sum_radii": round(float(np.percentile(
            separation/np.maximum(radii, 1.),5)),3),
        "first_site_tracks": "|".join(map(str, sorted(set(first.track_id.astype(int))))),
        "second_site_tracks": "|".join(map(str, sorted(set(second.track_id.astype(int))))),
        "first_site_center_x": round(float(first.x.median()),1),
        "first_site_center_y": round(float(first.y.median()),1),
        "second_site_center_x": round(float(second.x.median()),1),
        "second_site_center_y": round(float(second.y.median()),1),
        "disc_area_pixel_ceiling": int(np.rint(np.pi*(
            np.square(first_wrong.radius_px).sum()
            +np.square(second.radius_px).sum()))),
    }


def discover(labels: np.ndarray, points: pd.DataFrame) -> list[dict]:
    """Find spatially separate long paths, bridging tracker ID restarts."""
    if labels.ndim != 3:
        raise ValueError("expected a movie of two-dimensional labels")
    strong = points[points.state.eq("observed") & points.strong.astype(bool)].copy()
    t = strong.frame.to_numpy(dtype=int)
    y = np.clip(np.rint(strong.y).astype(int), 0, labels.shape[1]-1)
    x = np.clip(np.rint(strong.x).astype(int), 0, labels.shape[2]-1)
    strong["owner"] = labels[t, y, x].astype(int)
    by_frame = {int(frame): group for frame, group in strong.groupby("frame")}
    sites_cache: dict[tuple[int, int], pd.DataFrame] = {}
    found = []
    seen = set()
    for t0 in range(min(9, len(labels)-1)):
        start = by_frame.get(t0)
        if start is None:
            continue
        for name, named in start[start.owner > 0].groupby("owner"):
            if len(named) != 1:
                continue
            seed_a = named.iloc[0]
            for t1 in range(t0+1, min(t0+6, len(labels))):
                future = by_frame.get(t1)
                if future is None:
                    continue
                others = future[future.owner.eq(name)].copy()
                others["distance"] = np.hypot(others.x-float(seed_a.x),
                                                others.y-float(seed_a.y))
                others = others[others.distance > 3*(others.radius_px
                                         +float(seed_a.radius_px))]
                for _, seed_b in others.iterrows():
                    key_a = (t0, int(seed_a.track_id))
                    key_b = (t1, int(seed_b.track_id))
                    if key_a not in sites_cache:
                        sites_cache[key_a] = _site(seed_a, by_frame, len(labels))
                    if key_b not in sites_cache:
                        sites_cache[key_b] = _site(seed_b, by_frame, len(labels))
                    first, second = sites_cache[key_a], sites_cache[key_b]
                    summary = _candidate(first, second, int(name), strong)
                    if summary is None:
                        continue
                    signature = (int(name), tuple(sorted(set(first.track_id))),
                                 tuple(sorted(set(second.track_id))))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    found.append({"summary": summary, "first": first,
                                  "second": second})
    found.sort(key=lambda row: row["summary"]["disc_area_pixel_ceiling"],
               reverse=True)
    return found


def _path_components(labels: np.ndarray, site: pd.DataFrame,
                     strong_by_frame: dict[int, pd.DataFrame],
                     raw: np.ndarray | None = None) -> tuple[list[dict], str]:
    """Resolve a physical site to one existing labelled component per frame."""
    site = site.drop_duplicates("frame").sort_values("frame")
    frames = site.frame.to_numpy(dtype=int)
    records = []
    for t in range(int(frames.min()), int(frames.max())+1):
        x = float(np.interp(t, frames, site.x.to_numpy(float)))
        y = float(np.interp(t, frames, site.y.to_numpy(float)))
        radius = float(np.interp(t, frames, site.radius_px.to_numpy(float)))
        x0, y0 = int(round(x)), int(round(y))
        near = max(2, int(np.ceil(.7*radius)))
        ylo, yhi = max(0,y0-near), min(labels.shape[1],y0+near+1)
        xlo, xhi = max(0,x0-near), min(labels.shape[2],x0+near+1)
        patch = labels[t,ylo:yhi,xlo:xhi]
        yy, xx = np.where(patch > 0)
        if not len(xx):
            continue
        distances = np.hypot(xx+xlo-x, yy+ylo-y)
        nearest = int(np.argmin(distances))
        if distances[nearest] > max(2., .7*radius):
            continue
        py, px = int(yy[nearest]+ylo), int(xx[nearest]+xlo)
        old_name = int(labels[t,py,px])
        parts, _ = ndi.label(labels[t] == old_name, STRUCTURE)
        mask = parts == parts[py,px]
        ay, ax = np.where(mask)
        if not len(ax) or np.hypot(float(ax.mean())-x,float(ay.mean())-y) > max(8.,2.*radius):
            return [], "component_center_far_from_soma_site"
        distinct = []
        for other in strong_by_frame.get(t, pd.DataFrame()).itertuples(index=False):
            ox, oy = int(round(other.x)), int(round(other.y))
            if (0 <= ox < labels.shape[2] and 0 <= oy < labels.shape[1]
                    and mask[oy,ox] and np.hypot(float(other.x)-x,float(other.y)-y)
                    > 1.5*(radius+float(other.radius_px))):
                distinct.append(other)
        if distinct:
            if raw is None:
                return [], ("component_contains_distinct_strong_soma:"
                            f"frame={t+1}")
            soma = SimpleNamespace(x=x,y=y,radius_px=radius)
            basin = terminal._holder_basin(mask,raw[t],soma,distinct,
                                            float(mask.sum()),terminal.Params())
            if basin is None:
                return [], f"raw_soma_basin_unsupported:frame={t+1}"
            mask = basin
        records.append({"frame": t, "mask": mask, "old_name": old_name,
                        "area": int(mask.sum()), "x": x, "y": y,
                        "radius": radius})
    return records, "supported"


def correct(labels: np.ndarray, raw: np.ndarray, points: pd.DataFrame,
            protected: np.ndarray
            ) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if labels.shape != protected.shape or labels.shape != raw.shape:
        raise ValueError("raw and protected movies must match labels")
    candidate = labels.copy()
    occupied = np.zeros_like(labels, dtype=bool)
    next_name = int(labels.max())+1
    strong = points[points.state.eq("observed") & points.strong.astype(bool)].copy()
    strong_by_frame = {int(t): group for t,group in strong.groupby("frame")}
    audits, applications = [], []
    for proposal in discover(labels, points):
        summary = proposal["summary"]
        original = int(summary["original_name"])
        first, reason_a = _path_components(labels,proposal["first"],strong_by_frame)
        second, reason_b = _path_components(labels,proposal["second"],strong_by_frame,raw)
        reason = reason_a if reason_a != "supported" else reason_b
        if reason == "supported" and (len(first) < 60 or len(second) < 60):
            reason = "incomplete_labelled_soma_paths"
        if reason == "supported" and next_name > np.iinfo(labels.dtype).max:
            reason = "no_identity_values_remain"
        # A connected, unclaimed satellite of the second body must follow its
        # identity too; otherwise the original name would linger as a fragment.
        extra = []
        if reason == "supported":
            for record in second:
                if record["old_name"] != original:
                    continue
                t = record["frame"]
                parts, count = ndi.label(labels[t] == original,STRUCTURE)
                for part in range(1,count+1):
                    mask = parts == part
                    if np.any(mask & record["mask"]):
                        continue
                    yy, xx = np.where(mask)
                    if not len(xx):
                        continue
                    distance = float(np.hypot(xx.mean()-record["x"],
                                              yy.mean()-record["y"]))
                    scale = record["radius"]+np.sqrt(len(xx)/np.pi)
                    if (distance > 3.*scale or len(xx) > .4*record["area"]):
                        continue
                    if any(0 <= round(q.x) < labels.shape[2]
                           and 0 <= round(q.y) < labels.shape[1]
                           and mask[round(q.y),round(q.x)]
                           for q in strong_by_frame.get(t,pd.DataFrame()).itertuples(index=False)):
                        continue
                    extra.append({"frame":t,"mask":mask,"old_name":original,
                                  "area":int(len(xx)),"x":float(xx.mean()),
                                  "y":float(yy.mean()),"radius":float(np.sqrt(len(xx)/np.pi))})
            all_records = first+second+extra
            for record in all_records:
                t, mask = record["frame"],record["mask"]
                if np.any(mask & (protected[t] | occupied[t])):
                    reason = "crosses_protected_or_other_proposal"
                    break
            if reason == "supported":
                # A target component must never overlap the second soma body.
                for a in first:
                    for b in second:
                        if a["frame"] == b["frame"] and np.any(a["mask"] & b["mask"]):
                            reason = "two_soma_sites_share_one_component"
                            break
                    if reason != "supported":
                        break
        audits.append({**summary,"outcome":"applied" if reason=="supported" else "rejected",
                       "reason":reason,"first_resolved_frames":len(first),
                       "second_resolved_frames":len(second),"satellite_frames":len(extra)})
        if reason != "supported":
            continue
        for kind,records,new_name in (("restore_first_soma",first,original),
                                      ("separate_second_soma",second+extra,next_name)):
            for record in records:
                t,mask = record["frame"],record["mask"]
                if int(record["old_name"]) == new_name:
                    continue
                candidate[t][mask] = new_name
                occupied[t][mask] = True
                applications.append({"kind":kind,"frame":t+1,
                                     "old_name":int(record["old_name"]),
                                     "new_name":new_name,
                                     "changed_pixels":int(record["area"]),
                                     "original_name":original})
        next_name += 1
    changed = candidate != labels
    if (np.any(changed & protected)
            or not np.array_equal(candidate>0,labels>0)
            or int(changed.sum()) != sum(row["changed_pixels"] for row in applications)):
        raise AssertionError("persistent duplicate rule crossed accepted boundary")
    return candidate,pd.DataFrame(audits),pd.DataFrame(applications)


def main() -> None:
    base = json.loads((TUNING/"accepted_base.json").read_text(encoding="utf-8"))
    if base["attempt"] != "R119-A002":
        raise AssertionError("screen requires accepted R119 base")
    parent = json.loads((ROOT/base["accepted_manifest"]).read_text(encoding="utf-8"))
    paths = json.loads((TUNING/"inputs/paths.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for well in paths["base"]:
        labels = tifffile.imread(ROOT/parent["well_outputs"][well]["labels"])
        points = pd.read_csv(ROOT/paths["points"][well])
        points = points[points.state.eq("observed") & points.strong.astype(bool)].copy()
        t = points.frame.to_numpy(dtype=int)
        y = np.clip(np.rint(points.y).astype(int),0,labels.shape[1]-1)
        x = np.clip(np.rint(points.x).astype(int),0,labels.shape[2]-1)
        points["owner"] = labels[t,y,x].astype(int)
        by_frame = {int(frame): group for frame,group in points.groupby("frame")}
        sites_cache = {}
        for t0 in range(min(9,len(labels)-1)):
            start = by_frame.get(t0)
            if start is None:
                continue
            for name, named in start[start.owner>0].groupby("owner"):
                if len(named)!=1:
                    continue
                seed_a = named.iloc[0]
                for t1 in range(t0+1,min(t0+6,len(labels))):
                    future = by_frame.get(t1)
                    if future is None:
                        continue
                    others = future[future.owner.eq(name)].copy()
                    others["distance"] = np.hypot(others.x-float(seed_a.x),
                                                   others.y-float(seed_a.y))
                    others = others[others.distance > 3*(others.radius_px
                                      +float(seed_a.radius_px))]
                    for _, seed_b in others.iterrows():
                        key_a = (t0,int(seed_a.track_id))
                        key_b = (t1,int(seed_b.track_id))
                        if key_a not in sites_cache:
                            sites_cache[key_a] = _site(seed_a,by_frame,len(labels))
                        if key_b not in sites_cache:
                            sites_cache[key_b] = _site(seed_b,by_frame,len(labels))
                        candidate = _candidate(sites_cache[key_a],sites_cache[key_b],
                                               int(name),points)
                        if candidate:
                            rows.append({"well":well,**candidate})
        print("screened",well,flush=True)
    result = pd.DataFrame(rows)
    if len(result):
        result.drop_duplicates(["well","original_name","first_site_center_x",
                                "first_site_center_y","second_site_center_x",
                                "second_site_center_y"],inplace=True)
        result.sort_values("disc_area_pixel_ceiling",ascending=False,inplace=True)
    result.to_csv(OUT/"candidates.csv",index=False)
    print(result.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
