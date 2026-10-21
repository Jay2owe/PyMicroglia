"""Stream all camera frames with saved identity outlines to TIFF and MP4.

Display only: no tracking, filtering, cropping, resampling or measurement fits.
Memory is bounded by a few camera frames, independently of recording length.
"""
from contextlib import ExitStack
import colorsys
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import tifffile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def overlay(raw, labels, high, title):
    """Native pixels plus a header; label numbers and colours are display only."""
    gray = (np.clip(np.nan_to_num(raw.astype(np.float32)) / max(high, 1.), 0, 1) ** .7 * 180).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    edge = np.zeros(labels.shape, bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge &= labels > 0
    palette = np.array([colorsys.hsv_to_rgb((i * .61803398875) % 1, .65, 1.) for i in range(int(labels.max()) + 1)])
    palette = (palette * 255).astype(np.uint8)
    rgb[edge] = palette[labels[edge]]
    frame = Image.new('RGB', (raw.shape[1], raw.shape[0] + 32), 'black')
    frame.paste(Image.fromarray(rgb), (0, 32))
    pen = ImageDraw.Draw(frame)
    font = ImageFont.load_default(size=14)
    pen.text((8, 6), title, font=font, fill='white')
    for identity, box in enumerate(ndi.find_objects(labels), 1):
        if box is None: continue
        y, x = np.nonzero(labels[box] == identity)
        if not len(y): continue
        point = (int(x.mean()) + box[1].start, int(y.mean()) + box[0].start + 32)
        pen.text(point, str(identity), font=font, fill=tuple(palette[identity]),
                 stroke_width=1, stroke_fill='black', anchor='mm')
    return np.asarray(frame)


def render(measurement_dir, output_dir, *, prefix, fps=4., progress=None,
           registered_manifest=None, tracking_labels=None, tracking_offset=0):
    """Return verified full-duration files, reusing only matching complete output."""
    from auto_organotypic.video.encode import write_video
    import imageio_ffmpeg
    measurement_dir, output_dir = Path(measurement_dir), Path(output_dir)
    manifest_path = measurement_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    inputs = {row['role']: row for row in manifest['inputs']}
    channels = manifest['channels']
    count, offset = int(manifest['source_frames']), int(manifest['source_frame_offset'])
    coordinate_space = 'original camera'
    registration = None
    if registered_manifest is not None:
        registered_manifest = Path(registered_manifest)
        registration = json.loads(registered_manifest.read_text())
        if 'registered_crop' not in registration or registration['frames'] != count or tracking_labels is None:
            raise ValueError('Registered review requires the approved crop and matching tracking labels')
        roles = {row['role']: row for row in registration['outputs']}
        inputs = {'labels': dict(role='labels', path=str(tracking_labels), sha256=digest(tracking_labels))}
        for channel in channels:
            row = roles['detection_' + channel.lower()]
            inputs['channel:'+channel] = {**row, 'role': 'channel:'+channel}
        offset = int(tracking_offset)
        coordinate_space = 'registered crop'
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / f'{prefix}_full_recording.json'
    signature = dict(manifest_sha256=digest(manifest_path), producer_sha256=digest(Path(__file__)), fps=fps)
    if registration is not None:
        signature.update(registered_manifest_sha256=digest(registered_manifest),
                         tracking_labels_sha256=inputs['labels']['sha256'], tracking_offset=offset)
    if receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if saved['signature'] != signature: raise ValueError('Completed recording review uses different inputs or code')
        if any(digest(output_dir / name) != fingerprint for name, fingerprint in saved['files'].items()):
            raise ValueError('Completed recording review file changed')
        return saved
    times = pd.read_csv(measurement_dir/'cell_frame.csv', usecols=['source_frame_index', 'hours']).drop_duplicates()
    if times.source_frame_index.duplicated().any(): raise ValueError('Conflicting acquired timestamps')
    times = times.set_index('source_frame_index').hours.reindex(range(count))
    if times.isna().any(): raise ValueError('Missing acquired timestamp')
    with ExitStack() as stack:
        raw_files = {c: stack.enter_context(tifffile.TiffFile(inputs['channel:' + c]['path'])) for c in channels}
        labels_file = stack.enter_context(tifffile.TiffFile(inputs['labels']['path']))
        shape = raw_files[channels[0]].pages[0].shape
        if registration is not None and list(shape) != registration['canvas_shape'][1:]:
            raise ValueError('Review images do not match the approved registered crop')
        if len(shape) != 2 or any(len(f.pages) != count or f.pages[0].shape != shape for f in raw_files.values()):
            raise ValueError('Channels must have matching full camera frames')
        if len(labels_file.pages) + offset != count or labels_file.pages[0].shape != shape:
            raise ValueError('Saved outlines do not cover the acquisition clock and field')
        required = count * (shape[0]+32) * shape[1] * 3 * (len(channels)+1) + 256*1024**2
        if shutil.disk_usage(output_dir).free < required: raise OSError('Insufficient space for full recording review')
        for row in [inputs['labels'], *(inputs['channel:'+c] for c in channels)]:
            if digest(row['path']) != row['sha256']: raise ValueError('Review source changed: '+row['role'])
        high = {}
        for channel, opened in raw_files.items():
            sample = np.concatenate([opened.pages[int(i)].asarray()[::16, ::16].ravel()
                                     for i in np.unique(np.linspace(0, count-1, min(count, 41), dtype=int))])
            positive = sample[np.isfinite(sample) & (sample > 0)]
            high[channel] = float(np.percentile(positive, 99.5)) if positive.size else 1.
        names = {c: f'{prefix}_{c.lower()}_outlines.tif' for c in channels}
        temporary = {c: output_dir/(names[c]+'.partial') for c in channels}
        writers = {c: stack.enter_context(tifffile.TiffWriter(temporary[c], bigtiff=True)) for c in channels}
        movie = output_dir/f'{prefix}_full_recording.mp4'
        def frames():
            for index in range(count):
                labels = (np.zeros(shape, np.uint16) if index < offset else labels_file.pages[index-offset].asarray())
                panels = []
                for channel, opened in raw_files.items():
                    rendered = overlay(opened.pages[index].asarray(), labels, high[channel],
                        f'{prefix} | {channel} | Frame {index+1}/{count} | {times.iloc[index]:.2f} h')
                    writers[channel].write(rendered, photometric='rgb', compression='zlib',
                                           metadata={'axes': 'YXS'} if index == 0 else None)
                    panels.append(rendered)
                if progress: progress(index+1, count)
                joined = np.concatenate(panels, axis=1)
                # H.264 requires even dimensions: pad outside the preserved field.
                yield np.pad(joined, ((0, joined.shape[0] % 2), (0, joined.shape[1] % 2), (0, 0)))
        write_video(movie, frames(), fps=fps, profile='fast', extra_params=('-threads', '2'))
    for c in channels:
        with tifffile.TiffFile(temporary[c]) as opened:
            if len(opened.pages) != count: raise ValueError('Incomplete outline TIFF')
        temporary[c].replace(output_dir/names[c])
    decoded, _ = imageio_ffmpeg.count_frames_and_secs(str(movie))
    if decoded != count: raise ValueError('Video frame count differs from the acquired recording')
    files = {name: digest(output_dir/name) for name in [movie.name, *names.values()]}
    result = dict(status='complete', signature=signature, recording=prefix,
        channels=channels, frames=count, decoded_frames=decoded, camera_shape=list(shape),
        coordinate_space=coordinate_space, registered_crop=registration.get('registered_crop') if registration else None,
        source_frame_offset=offset, source_hours=[float(times.iloc[0]), float(times.iloc[-1])],
        native_pixels_preserved=True, header_height=32, display_high=high,
        display_only=True, individual_cell_videos=False, files=files,
        source_manifest=str(manifest_path), source_inputs=list(inputs.values()),
        registered_manifest=str(registered_manifest) if registered_manifest is not None else None)
    receipt_path.write_text(json.dumps(result, indent=2))
    return result
