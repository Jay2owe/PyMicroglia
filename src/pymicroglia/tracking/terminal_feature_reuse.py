"""Reuse immutable terminal-tail measurements, retaining original decision code.

Only the source-verified feature-building block is replaced. Unknown scientific
versions use their original implementation. State lives for one proposal call.
"""
from contextlib import contextmanager
import hashlib
import inspect
import math
import numpy as np

SOURCE_SHA256 = '1bdd1837ef24609c9681c3d2eac561d9057d84175a4fcb82f9a1745532cd72d0'


class Feature:
    def __init__(self, coordinates, width, pure, strong, component, statistics):
        self.centroid = np.mean(coordinates, axis=0)
        self.area = np.int64(len(coordinates))
        self.flat = coordinates[:, 0] * width + coordinates[:, 1]
        self.pure, self.strong, self.component = pure, strong, component
        statistics['feature_bytes'] += self.flat.nbytes + self.centroid.nbytes

    def __and__(self, other):
        # Both vectors contain unique, ascending row-major pixel indices, so
        # the shorter one's values are looked up in the longer one by binary
        # search instead of sorting both together.
        small, large = (self.flat, other.flat) if len(self.flat) <= len(other.flat) else (other.flat, self.flat)
        if not small.size:
            return False
        at = np.searchsorted(large, small)
        at[at == large.size] = large.size - 1
        return bool(np.any(large[at] == small))


def mask_step(left, right):
    if not left.area or not right.area:
        return float('inf')
    radius = math.sqrt(max(
        ((left.area / math.pi) + (right.area / math.pi)) / 2.0,
        np.finfo(float).eps))
    return float(np.linalg.norm(left.centroid - right.centroid)) / radius


class Features:
    def __init__(self, labels, visible, module, counters):
        self.labels, self.module, self.counters = labels, module, counters
        self.frames, self.values = {}, {}
        frame = visible.frame.astype(int).to_numpy()
        owner = visible.candidate_owner.astype(int).to_numpy()
        strong = visible.strong.astype(bool).to_numpy()
        self.strong = set(zip(frame[strong], owner[strong]))

    def get(self, frame, owner):
        key = (frame, owner)
        if key in self.values:
            self.counters['feature_hits'] += 1
            return self.values[key]
        self.counters['feature_misses'] += 1
        components = None
        if frame not in self.frames:
            components, owners, _ = self.module.component_accounting.label_value_components(self.labels[frame])
            self.counters['frame_builds'] += 1
            boxes = self.module.ndi.find_objects(components)
            first = np.full(len(owners) + 1, components.size, dtype=np.intp)
            np.minimum.at(first, components.ravel(), np.arange(components.size, dtype=np.intp))
            hosts, _ = self.module.ndi.label(self.labels[frame] > 0, np.ones((3, 3), dtype=np.uint8))
            component_hosts = hosts.ravel()[first[1:]] if len(owners) else []
            host_owners = {}
            owner_components = {}
            for component, (identity, host) in enumerate(zip(owners, component_hosts), start=1):
                host_owners.setdefault(int(host), set()).add(int(identity))
                owner_components.setdefault(int(identity), []).append(component)
            descriptors = {}
            for identity, indices in owner_components.items():
                if len(indices) == 1:
                    component = indices[0]
                    pure = host_owners[int(component_hosts[component - 1])] == {identity}
                    descriptors[identity] = (component, boxes[component - 1], pure)
            self.frames[frame] = descriptors
        descriptor = self.frames[frame].get(owner)
        if descriptor is None:
            value = None
        else:
            component, box, pure = descriptor
            if components is None:
                # Only a described owner needs the component map again.
                components = self.module.component_accounting.label_value_components(self.labels[frame])[0]
            yy, xx = np.nonzero(components[box] == component)
            coordinates = np.column_stack((yy + box[0].start, xx + box[1].start))
            value = Feature(coordinates, self.labels.shape[2], pure, key in self.strong,
                            component, self.counters)
        self.values[key] = value
        return value


@contextmanager
def install(module):
    original = module._terminal_successor_proposals
    source = inspect.getsource(original).rstrip()
    counters = dict(feature_hits=0, feature_misses=0, frame_builds=0,
                    feature_bytes=0, source_supported=False)
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        yield counters
        return
    counters['source_supported'] = True
    start = source.index('            components, owners, _ =')
    end_marker = '            strong_frames += int(frame_points.strong.astype(bool).any())'
    end = source.index(end_marker, start) + len(end_marker)
    replacement = '''            feature = measured.get(frame, source)
            if feature is None:
                continue
            keys.append(ComponentKey(frame, feature.component))
            masks.append(feature)
            pure_hosts += int(feature.pure)
            strong_frames += int(feature.strong)'''
    source = source[:start] + replacement + source[end:]
    source = source.replace('    seeds: list[dict] = []',
                            '    measured = _make_features(labels, visible)\n    seeds: list[dict] = []', 1)
    namespace = dict(original.__globals__)
    namespace['_make_features'] = lambda labels, visible: Features(labels, visible, module, counters)
    namespace['_mask_step'] = mask_step
    exec(compile(source, __file__, 'exec'), namespace)
    module._terminal_successor_proposals = namespace[original.__name__]
    try:
        yield counters
    finally:
        module._terminal_successor_proposals = original
