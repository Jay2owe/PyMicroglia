"""Check unchanged vector marks and native nearest-neighbour image colour cells."""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

SVG = '{http://www.w3.org/2000/svg}'
HREF = '{http://www.w3.org/1999/xlink}href'
GEOMETRY = {'x', 'y', 'x1', 'x2', 'y1', 'y2', 'width', 'height', 'cx', 'cy',
            'r', 'rx', 'ry', 'd', 'points', 'transform', 'viewBox'}
NUMBER = re.compile(r'[-+]?(?:\d*\.?\d+)(?:[eE][-+]?\d+)?')


def vectors_equivalent(left, right):
    """Allow only the last decimal-place rounding in SVG coordinates."""
    if type(left) is not type(right):
        return False
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(vectors_equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return False
        for key, a in left.items():
            b = right[key]
            if a == b:
                continue
            if key not in GEOMETRY or NUMBER.sub('#', a) != NUMBER.sub('#', b):
                return False
            x, y = NUMBER.findall(a), NUMBER.findall(b)
            if len(x) != len(y) or any(abs(float(u)-float(v)) > 1.01e-6 for u, v in zip(x, y)):
                return False
        return True
    return left == right


def matrix_for(attrs):
    transform = np.eye(3)
    text = attrs.get('transform', '')
    for name, arguments in re.findall(r'(\w+)\(([^)]*)\)', text):
        values = [float(v) for v in re.split(r'[,\s]+', arguments.strip()) if v]
        matrix = np.eye(3)
        if name == 'matrix' and len(values) == 6:
            a, b, c, d, e, f = values
            matrix = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        elif name == 'scale':
            matrix[0, 0] = values[0]
            matrix[1, 1] = values[-1]
        elif name == 'translate':
            matrix[:2, 2] = [values[0], values[1] if len(values) == 2 else 0]
        else:
            raise ValueError('Unverified SVG image transform: '+text)
        transform = transform @ matrix
    return transform


def bounds(attrs):
    transform = matrix_for(attrs)
    x, y = float(attrs.get('x', 0)), float(attrs.get('y', 0))
    w, h = float(attrs['width']), float(attrs['height'])
    corners = (transform @ np.array([[x, y, 1], [x+w, y, 1], [x, y+h, 1], [x+w, y+h, 1]]).T).T
    return np.array([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])

def intersection(a, b):
    return np.r_[np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])]


def visible_bounds(attrs):
    box = bounds(attrs)
    clip = attrs.get('_clip_bounds')
    return box if clip is None else intersection(box, np.asarray(clip))


def image_style(attrs):
    result = {k: v for k, v in attrs.items() if k not in {
        'id', 'transform', 'x', 'y', 'width', 'height', '_clip_bounds'}}
    style = ';'.join(part for part in result.pop('style', '').split(';')
                     if part.strip() and not part.strip().startswith('image-rendering:'))
    if style:
        result['style'] = style
    return result


def local_raster_match(a, b):
    """Match unchanged raster boundaries within one original export pixel."""
    def covered(left, right):
        padded = np.pad(right.astype(int), ((1, 1), (1, 1), (0, 0)), mode='edge')
        found = np.zeros(left.shape[:2], dtype=bool)
        left = left.astype(int)
        for dy in range(3):
            for dx in range(3):
                nearby = padded[dy:dy+left.shape[0], dx:dx+left.shape[1]]
                found |= ((np.max(np.abs(left[..., :3]-nearby[..., :3]), axis=2) <= 1)
                          & (left[..., 3] == nearby[..., 3]))
        return bool(found.all())
    return covered(a, b) and covered(b, a)


def native_colours(old_pixels, new_pixels, old_attrs, new_attrs):
    old_box, new_box = bounds(old_attrs), bounds(new_attrs)
    visible = intersection(visible_bounds(old_attrs), visible_bounds(new_attrs))
    old = oriented_pixels(old_pixels, old_attrs).astype(int)
    new = oriented_pixels(new_pixels, new_attrs).astype(int)
    indices, samples = [], []
    for dim, count in [(0, new_pixels.width), (1, new_pixels.height)]:
        edges = np.linspace(new_box[dim], new_box[dim+2], count+1)
        low = np.maximum(edges[:-1], visible[dim])
        high = np.minimum(edges[1:], visible[dim+2])
        selected = np.flatnonzero(high-low > 1e-8)
        indices.append(selected)
        samples.append((low[selected]+high[selected])/2)
    if any(not len(index) for index in indices):
        return False
    x = np.clip(((samples[0]-old_box[0])/(old_box[2]-old_box[0])*old_pixels.width).astype(int), 0, old_pixels.width-1)
    y = np.clip(((samples[1]-old_box[1])/(old_box[3]-old_box[1])*old_pixels.height).astype(int), 0, old_pixels.height-1)
    a, b = old[np.ix_(y, x)], new[np.ix_(indices[1], indices[0])]
    return bool(np.array_equal(a[..., 3], b[..., 3]) and np.max(np.abs(a[..., :3]-b[..., :3])) <= 1)


def oriented_pixels(pixels, attrs):
    """Compare displayed rows, accounting for SVG's image-coordinate flips."""
    matrix = matrix_for(attrs)
    if abs(matrix[0, 1]) > 1e-10 or abs(matrix[1, 0]) > 1e-10:
        raise ValueError('Rotated image needs separate rendered-image verification')
    values = np.asarray(pixels)
    if matrix[0, 0] < 0:
        values = values[:, ::-1]
    if matrix[1, 1] < 0:
        values = values[::-1]
    return values


def tree(path):
    root = ET.parse(path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == SVG+'metadata':
                parent.remove(child)
    identifiers = {node.attrib['id']: 'id'+str(i) for i, node in enumerate(root.iter())
                   if 'id' in node.attrib}
    clips = {node.attrib['id']: bounds(dict(node[0].attrib)) for node in root.iter(SVG+'clipPath')
             if len(node) == 1 and node[0].tag == SVG+'rect'}
    images = []

    def normalize(node, active_clip=None):
        reference = re.fullmatch(r'url\(#([^)]+)\)', node.attrib.get('clip-path', ''))
        if reference and reference[1] in clips:
            own_clip = clips[reference[1]]
            active_clip = own_clip if active_clip is None else intersection(active_clip, own_clip)
        if node.tag == SVG+'image':
            attrs = dict(node.attrib)
            if active_clip is not None:
                attrs['_clip_bounds'] = active_clip.tolist()
            blob = base64.b64decode(attrs.pop(HREF).split(',')[1])
            pixels = Image.open(BytesIO(blob)).convert('RGBA')
            images.append((attrs, pixels))
            return ('image',)
        attrs = {}
        for key, value in node.attrib.items():
            if key == 'id':
                value = identifiers[value]
            elif value.startswith('#'):
                value = '#'+identifiers.get(value[1:], value[1:])
            else:
                value = re.sub(r'url\(#([^)]+)\)', lambda m: 'url(#'+identifiers.get(m[1], m[1])+')', value)
            attrs[key] = value
        return node.tag, attrs, (node.text or '').strip(), [normalize(c, active_clip) for c in node]
    return normalize(root), images


def compare(left, right):
    a, ai = tree(left)
    b, bi = tree(right)
    checks = []
    for (attrs_a, pixels_a), (attrs_b, pixels_b) in zip(ai, bi):
        box_a, box_b = visible_bounds(attrs_a), visible_bounds(attrs_b)
        # The exported pixel pitch belongs to the full raster. Clipping its
        # last fraction of a pixel must not shrink the rounding allowance.
        raster_box = bounds(attrs_a)
        pixel_size = max((raster_box[2]-raster_box[0])/pixels_a.width,
                         (raster_box[3]-raster_box[1])/pixels_a.height)
        edge_change = float(np.max(np.abs(box_a-box_b)))
        same_placement = edge_change <= pixel_size + 1e-5
        if pixels_a.size == pixels_b.size:
            exact = np.array_equal(oriented_pixels(pixels_a, attrs_a), oriented_pixels(pixels_b, attrs_b))
            method = 'exact decoded image pixels'
        elif 'pixelated' in attrs_b.get('style', ''):
            # Native colour mapping avoids the floating resampling round-trip.
            # At a colormap bin boundary it can move by one 8-bit colour step.
            # Scientific tables remain exact; opacity remains exact.
            exact = native_colours(pixels_a, pixels_b, attrs_a, attrs_b)
            method = 'every visible native cell within one 8-bit RGB step; exact alpha; unchanged vector clipping'
        elif max(abs(pixels_a.width-pixels_b.width), abs(pixels_a.height-pixels_b.height)) <= 1:
            sampled = pixels_a.resize(pixels_b.size, resample=Image.Resampling.NEAREST)
            exact = local_raster_match(oriented_pixels(sampled, attrs_a), oriented_pixels(pixels_b, attrs_b))
            method = 'unchanged raster renderer: one-pixel dimension/boundary rounding, also checked on baseline repeats'
        else:
            exact = False
            method = 'unexplained image size change'
        checks.append({'before_size': pixels_a.size, 'after_size': pixels_b.size,
                       'equal': bool(exact and same_placement and image_style(attrs_a) == image_style(attrs_b)),
                       'method': method, 'image_style_equal': image_style(attrs_a) == image_style(attrs_b),
                       'max_edge_change_points': edge_change,
                       'one_original_output_pixel_points': pixel_size})
    vectors = vectors_equivalent(a, b)
    return {'vectors_equal': a == b, 'vectors_equivalent': vectors,
            'coordinate_tolerance_points': 1.01e-6, 'same_image_count': len(ai) == len(bi),
            'images': checks, 'parity': vectors and len(ai) == len(bi) and all(c['equal'] for c in checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--actions', nargs='+')
    args = parser.parse_args()
    summary = json.loads((args.before/'summary.json').read_text())
    rows = []
    for name in args.actions or summary['required_actions']:
        receipts = [json.loads((folder/(name+'.json')).read_text()) for folder in (args.before, args.after)]
        if len(receipts[0].get('files', [])) != len(receipts[1].get('files', [])):
            rows.append({'action': name, 'parity': False, 'reason': 'different export counts'})
            continue
        for left, right in zip(receipts[0].get('files', []), receipts[1].get('files', [])):
            if Path(left).suffix != '.svg':
                continue
            row = {'action': name, 'files': [left, right], **compare(left, right)}
            rows.append(row)
            print(name, row['parity'], flush=True)
    args.output.write_text(json.dumps({'cases': rows, 'parity': bool(rows) and all(r['parity'] for r in rows)},
                                         indent=2), encoding='utf-8')
    return 0 if rows and all(r['parity'] for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
