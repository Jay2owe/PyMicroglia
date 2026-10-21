"""Cell identifiers and condition inheritance shared by saved comparison views."""
from __future__ import annotations

import re


def parse_cell_ids(text):
    """Parse positive IDs and inclusive ranges, without expanding unbounded input."""
    if not isinstance(text, str) or len(text) > 20000:
        raise ValueError('Enter cell IDs such as 1, 5, 10-20')
    identities = set()
    for token in re.split(r'[,;\s]+', text.strip()):
        if not token:
            continue
        match = re.fullmatch(r'([0-9]+)(?:[-–]([0-9]+))?', token)
        if not match:
            raise ValueError('Enter cell IDs such as 1, 5, 10-20')
        start, end = int(match[1]), int(match[2] or match[1])
        if start < 1 or end < start or end > 2**53-1:
            raise ValueError('Cell IDs must be positive; ranges must run from smaller to larger')
        if end-start+1 > 100000 or len(identities) > 100000:
            raise ValueError('Select at most 100,000 cell IDs at a time')
        identities.update(range(start, end+1))
    if len(identities) > 100000:
        raise ValueError('Select at most 100,000 cell IDs at a time')
    return sorted(identities)


def validate_cell_conditions(assignments, conditions=None):
    if not isinstance(assignments, dict):
        raise ValueError('Cell conditions must map cell IDs to conditions')
    for identity, condition in assignments.items():
        if not isinstance(identity, str) or not re.fullmatch(r'[1-9][0-9]*', identity) or int(identity) > 2**53-1:
            raise ValueError('Cell conditions need positive whole-number cell IDs')
        if not isinstance(condition, str) or conditions is not None and condition and condition not in conditions:
            raise ValueError('Unknown condition assigned to a cell')
    return assignments


def cell_condition(source, identity, default='Unassigned'):
    """An explicit cell label (including Unassigned) overrides its recording."""
    return source.get('cell_conditions', {}).get(str(int(identity)), source.get('condition', default)) or 'Unassigned'


def frame_conditions(source, identities, default='Unassigned'):
    validate_cell_conditions(source.get('cell_conditions', {}))
    return identities.map(lambda identity: cell_condition(source, identity, default))
