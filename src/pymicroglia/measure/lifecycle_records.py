"""Verified lifecycle-review tables shared by native figures and the workbench."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

SCHEMA = 'pymicroglia.exploration-lifecycle.v1'
TABLES = ('lifecycle_events', 'lifecycle_cells')


def review_folder(root, stem):
    return root / 'AI_Exports/lifecycle_review' / hashlib.sha256(stem.encode()).hexdigest()[:16]


@lru_cache(maxsize=64)
def _content_hash(path, size, modified):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _content_hash(str(path), stat.st_size, stat.st_mtime_ns)


def review_scope(root, movie, measurement_path, labels_path):
    return {'movie': hashlib.sha256(json.dumps(movie, sort_keys=True).encode()).hexdigest(),
            'measurements': file_sha256(measurement_path), 'refined_labels': file_sha256(labels_path)}


def review_table(root, movie, name, *, measurement_path, labels_path):
    """Read classification from the original masks; reject changed inputs or tables."""
    index = review_folder(root, movie['stem']) / 'review.json'
    if not index.is_file():
        return None
    if index.stat().st_size > 1024**2:
        raise ValueError('The lifecycle review index exceeds the metadata budget')
    record = json.loads(index.read_text(encoding='utf-8'))
    if record.get('schema') != SCHEMA or record.get('scope') != review_scope(root, movie, measurement_path, labels_path):
        raise ValueError('The lifecycle review does not match these masks and measurements')
    for source in record['inputs'].values():
        source_path = Path(source['path'])
        source_path = source_path if source_path.is_absolute() else root / source_path
        if file_sha256(source_path) != source['sha256']:
            raise ValueError('Original tracking inputs changed after lifecycle review')
    table = record['tables'].get(name)
    if not table:
        return None
    path = (root / table['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('A lifecycle review table leaves the selected run')
    if file_sha256(path) != table['sha256']:
        raise ValueError('A lifecycle review table changed after classification')
    return path
