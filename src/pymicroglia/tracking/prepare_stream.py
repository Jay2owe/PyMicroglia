"""Bounded-memory IO for the unchanged six-array Motion input arithmetic."""
from __future__ import annotations
import contextlib,gc, hashlib, json, shutil, tempfile, zipfile
from importlib import resources
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from . import motion_preparation as evidence
from .prepare import TARGET_MEDIAN, sha256

METHOD_VERSION = "2026-10-09-motion-inputs-stream-v1"

def _masked_median(photons,cells,path):
    """Keep the exact float32 median without concatenating a movie in RAM."""
    count=sum(int(np.count_nonzero(mask>0)) for mask in cells)
    if not count:raise ValueError("the mask is empty, so there is nothing to track")
    values=np.memmap(path,mode='w+',dtype=np.float32,shape=(count,));start=0
    try:
        for frame,mask in zip(photons,cells):
            part=np.clip(frame[mask>0].astype(np.float32),0,None)
            values[start:start+len(part)]=part;start+=len(part)
            values.flush()
        assert start==count
        result=float(np.median(values,overwrite_input=True));values.flush();return result
    finally:
        values._mmap.close();Path(path).unlink(missing_ok=True)

def build(stem, photons, cells_path, folder, *, frame_interval_min, dataset, scratch_dir=None,
          save_measurement_raw=True, native_signal=False, progress=None, registered_raw_path=None):
    """Reuse per-transition arithmetic and on-disk arrays; preserve native photons."""
    photons = np.asarray(photons)
    if photons.ndim != 3 or len(photons) < 2:
        raise ValueError('Motion requires at least two (T,Y,X) frames')
    if native_signal:
        if cells_path is not None:
            raise ValueError('Native fluorescence Motion detection cannot be gated by upstream masks')
        if photons.dtype != np.uint16:
            raise ValueError('Native fluorescence input must be uint16 signal counts')
    elif registered_raw_path is not None:
        raise ValueError('A retained raw link is only valid for unchanged native fluorescence')
    else:
        cells_path = Path(cells_path)
        with tifffile.TiffFile(cells_path) as opened:
            if photons.shape != opened.series[0].shape:
                raise ValueError("photons and cells must share (T,Y,X)")
    target = Path(folder); target.mkdir(parents=True, exist_ok=True)
    # Scratch arrays occupy under 24 bytes per movie pixel; accepted files remain.
    required = int(photons.size * 24 + 2 * 1024**3)
    scratch_root=Path(scratch_dir) if scratch_dir is not None else target
    scratch_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(scratch_root).free < required:
        raise OSError(f"Motion preparation needs {required} free bytes including scratch")
    params = dict(evidence.DEFAULTS)
    shape = (len(photons)-1, *photons.shape[1:])
    with tempfile.TemporaryDirectory(prefix="motion-prepare-", dir=scratch_root) as scratch_name,contextlib.ExitStack() as mappings:
        scratch = Path(scratch_name); arrays = {}
        def release(array):
            if not array._mmap.closed:array._mmap.close()
        cells = None
        scale = 1.0
        if not native_signal:
            with tifffile.TiffFile(cells_path) as opened:
                cells=opened.series[0].asarray(out=str(scratch/'cells.dat'))
            mappings.callback(release,cells)
            arrays['cells']=cells;cells.flush()
            median=_masked_median(photons,cells,scratch/'median.dat')
            if median<=0:raise ValueError("median masked photon rate is zero")
            scale=TARGET_MEDIAN/median
        def allocate(role, dimensions, dtype):
            array = np.memmap(scratch/f"{role}.dat", mode="w+", shape=dimensions, dtype=dtype)
            mappings.callback(release,array)
            arrays[role] = array
            return array
        raw = allocate("registered_raw", photons.shape, np.uint16)
        ratio = allocate("lag_float", shape, np.float32)
        components = allocate("neutral_tracks", shape, np.uint16)
        trails = allocate("trail_labels", shape, np.uint16)
        ages = allocate("trail_ages", shape, np.uint8)
        composite = allocate("motion_composite", (shape[0], 5, *shape[1:]), np.uint16)
        gain = allocate("gain", shape, bool); loss = allocate("loss", shape, bool)
        for t, frame in enumerate(photons):
            if native_signal:
                raw[t] = frame
            else:
                inside = cells[t] > 0
                values = np.clip(frame.astype(np.float32), 0, None) * scale
                values[~inside] = 0
                raw[t] = np.rint(np.clip(values, 0, 65535)).astype(np.uint16)
                raw[t][inside & (raw[t] == 0)] = 1
            if (t+1)%16==0:raw.flush()
        previous_rows = []; flat = []; next_track = 1
        for t in range(shape[0]):
            ratio[t] = evidence.lag_ratio(raw[t:t+2])[0]
            gate = evidence.absolute_change_gate(raw[t:t+2], params["delta_noise_sigma"])[0]
            labels, rows, gained, lost = evidence.transition_components(
                raw[t], raw[t+1], ratio[t], gate,
                **{key: params[key] for key in ("neutral_log2", "lobe_log2", "min_core_px", "min_lobe_px", "lobe_reach_px")})
            mapping = {}
            if t:
                for old_index,new_index in evidence.link_component_rows(
                    previous_rows, rows, **{key:params[key] for key in ("max_link_px", "min_area_ratio", "max_area_ratio", "max_mean_log2")}):
                    mapping[new_index] = int(previous_rows[old_index]["track"])
            components[t] = 0
            for index,row in enumerate(rows):
                track = mapping.get(index)
                if track is None:
                    track = next_track; next_track += 1
                if track > np.iinfo(np.uint16).max:
                    raise ValueError("Motion breadcrumb identities exceed uint16 capacity")
                row.update(t=int(t), track=int(track))
                components[t][labels == int(row["component"])] = track
                flat.append({"imagej_from":t+1,"imagej_to":t+2,**row})
            gain[t] = gained; loss[t] = lost
            previous_rows = rows
            if (t+1)%16==0:
                for buffer in (ratio,components,gain,loss):buffer.flush()
            if progress is not None:
                progress('Motion evidence',t+1,shape[0])
        columns=("t","imagej_from","imagej_to","component","track","y","x","area_px","mean_intensity","gain_px_near","loss_px_near","motion_qualified")
        table=pd.DataFrame(flat,columns=columns)
        evidence.rolling_trails(components,table,params["trail_window"],output_arrays=(trails,ages))
        for t in range(shape[0]):
            channels,_,_=evidence.composite_channels(raw[t:t+2],ratio[t:t+1],gain[t:t+1],loss[t:t+1],components[t:t+1],ages[t:t+1],params)
            composite[t]=np.stack(channels,axis=1)[0]
            if (t+1)%16==0:composite.flush()
        names={"registered_raw":f"{stem}.tif","lag_float":f"{stem}_lag01_float.tif","neutral_tracks":f"{stem}_neutral_tracks.tif","trail_labels":f"{stem}_trail_labels.tif","trail_ages":f"{stem}_trail_ages.tif","motion_composite":f"{stem}_motion_evidence.tif"}
        pins={}
        for role,name in names.items():
            path=target/name
            array=arrays[role]
            if role=='registered_raw' and registered_raw_path is not None:
                import os
                source=Path(registered_raw_path)
                with tifffile.TiffFile(source) as opened:
                    if opened.series[0].shape!=photons.shape or opened.series[0].dtype!=photons.dtype:
                        raise ValueError('Retained native signal differs in shape or dtype')
                    for t,page in enumerate(opened.pages):
                        if not np.array_equal(page.asarray(),photons[t]):raise ValueError('Retained native signal differs in pixels')
                os.link(source,path)
            else:
                tifffile.imwrite(path,array,photometric="minisblack",compression="zlib",compressionargs={"level":1},metadata={"axes":"TCYX" if role=="motion_composite" else "TYX"})
            key="relative_to_registered_input_dir" if role=="registered_raw" else "relative_to_motion_input_dir"
            pins[role]={key:name,"sha256":sha256(path)}
        measurement_raw=target/f"{stem}_registered_photons.tif" if save_measurement_raw else None
        if measurement_raw is not None:
            tifffile.imwrite(measurement_raw,(frame.astype(np.float32) for frame in photons),shape=photons.shape,
                dtype=np.float32,photometric="minisblack",compression="zlib",compressionargs={"level":1},metadata={"axes":"TYX"})
        # Windows cannot remove mapped scratch files until every view is closed.
        for array in arrays.values():array.flush();array._mmap.close()
        arrays.clear()
        del raw,ratio,components,trails,ages,composite,gain,loss,array,cells
        gc.collect()
    with resources.files("pymicroglia.tracking").joinpath("data/motion_engine.zip").open("rb") as handle:
        with zipfile.ZipFile(handle) as archive:config=json.loads(archive.read("config.json"))
    config.update(dataset=dataset,stems=[stem],frame_interval_min=float(frame_interval_min),registered_input_dir=".",motion_input_dir=".",pinned_files={stem:pins})
    if native_signal:
        config['source_frame_offset'] = 0
        for section, key in (
                ('field_wide_owner_consensus','motion_frame_offset'),
                ('field_wide_bracketed_established_owner_invasion','raw_frame_offset')):
            values = config.get('accepted_postprocessing',{}).get(section,{})
            if section.endswith('invasion'):
                values = values.get('parameters',{})
            if key in values:
                values[key] = 0
    config_path=target/"motion_config.json";config_path.write_text(json.dumps(config,indent=2)+"\n")
    return dict(stem=stem,config=str(config_path),pins=pins,masked_photon_scale=scale,
        measurement_raw=str(measurement_raw) if measurement_raw is not None else None,
        measurement_raw_sha256=sha256(measurement_raw) if measurement_raw is not None else None,
        frame_interval_min=float(frame_interval_min),cells_sha256=sha256(cells_path) if cells_path is not None else None,
        detection_mode='native_motion' if native_signal else 'masked_photons',
        neutral_tracks=int(table.track.nunique()) if len(table) else 0,method_version=METHOD_VERSION)
