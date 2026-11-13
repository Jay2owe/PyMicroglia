"""Core-replacement ownership: point nodes assigned once, and next-frame owner nodes indexed.

preserve_core_replacement_owners set each point's node with one pts.loc write
per row; with a unique index the same int64 column is assigned once from the
collected values (a repeated index label keeps the per-row writes, where the
last write wins for every row of that label). root_owner scanned every node
record of the movie for the nodes one frame later that hold the owner;
records are only read in this function, so one (frame, identity) -> node set
index built at the first such question gives the same sets.

The point loop read only frame, y, x and radius_px of each iterrows row. Those
values came from DataFrame.values, which holds each column's numbers as the
Python scalars its tolist gives (or, for an all-numeric table, as float64);
int, np.floor and the loop's float arithmetic give the same results for either.
The loop takes them from four column lists instead of a Series per row. Tables
whose four columns are not unique NumPy columns keep iterrows.

The three per-track tables skip their frame sort when no track repeats or
reverses a frame in table order (track_sort_skips.tracks_in_order); a sort of
frames already strictly increasing within each group leaves them as they are.

A point off every node looked for the nearest node pixel in its disc by
building the window's full coordinate grids. Only the window's node pixels can
be hit, so the same distance expression is evaluated at those pixels alone
(same integer offsets, same point coordinates, same values); the minimum and
the unique node values among the nearest hits are the same.
"""
import hashlib
import inspect
from pathlib import Path
import sys
try:
    from . import track_sort_skips as _sort_skips
except ImportError:
    import track_sort_skips as _sort_skips

SOURCE = 'bc2f4e19a54ccaaa9a6b1263ef0af96af279f5a3b1282835906abfa4673f8235'
REPLACEMENTS = [
    ("""    pts['node']=0
    for index,p in pts.iterrows():
        f=int(p.frame);y=int(np.floor(p.y+.5));x=int(np.floor(p.x+.5))
        if not (0<=f<len(labels) and 0<=y<labels.shape[1] and 0<=x<labels.shape[2]):continue
""", """    pts['node']=0
    _unique_rows=pts.index.is_unique;_point_nodes=[]
    for index,p in _point_rows(pts):
        f=int(p.frame);y=int(np.floor(p.y+.5));x=int(np.floor(p.x+.5))
        if not (0<=f<len(labels) and 0<=y<labels.shape[1] and 0<=x<labels.shape[2]):
            _point_nodes.append(0);continue
"""),
    ("""        pts.loc[index,'node']=n
    pts['owner']=""", """        if _unique_rows:_point_nodes.append(n)
        else:pts.loc[index,'node']=n
    if _unique_rows and len(pts):pts['node']=np.asarray(_point_nodes,dtype=pts['node'].dtype)
    pts['owner']="""),
    ("""                field=nodes[int(final.frame)];held={q['node'] for q in records[1:]
                    if q['frame']==int(final.frame)+1 and q['identity']==owner}
""", """                field=nodes[int(final.frame)];held=_held_nodes(records,int(final.frame)+1,owner)
"""),
]


# The disc search of a point that is off every node, over the window's node pixels only.
DISC = ("""            field=nodes[f,y0:y1,x0:x1];yy,xx=np.indices(field.shape)
            distance=(yy+y0-p.y)**2+(xx+x0-p.x)**2
            hit=(field>0)&(distance<=radius**2)
            if hit.any():
                minimum=distance[hit].min();choices=np.unique(field[hit&(distance==minimum)])
                if len(choices)==1:n=int(choices[0])
""", """            n=_nearest_node(nodes[f,y0:y1,x0:x1],y0,x0,p.y,p.x,radius,n)
""")


def nearest_node(stats):
    import numpy as np

    def _nearest_node(field, y0, x0, py, px, radius, n):
        yy, xx = np.nonzero(field)
        stats['disc_points'] = stats.get('disc_points', 0) + 1
        if not len(yy):
            return n
        values = field[yy, xx]
        distance = (yy+y0-py)**2+(xx+x0-px)**2
        hit = distance <= radius**2
        if hit.any():
            minimum = distance[hit].min()
            choices = np.unique(values[hit&(distance==minimum)])
            if len(choices) == 1:
                n = int(choices[0])
        return n
    return _nearest_node


# A source's observation in the frame before a replacement: the last row of
# that frame, read from one frame -> last position index per track table.
PREVIOUS = ('previous=obs[obs.frame.eq(f-1)].iloc[-1]', 'previous=_previous_row(obs,f-1)')
# Per-track tables whose frame sort is skipped when no track repeats or
# reverses a frame in table order (track_sort_skips.tracks_in_order).
GROUPS = tuple(("    %s={int(k):g.sort_values('frame') for k,g in %s.groupby('track_id')}\n" % (name, table),
                "    %s=_tracks_in_order(%s,'track_id')\n" % (name, table))
               for name, table in (('by_track', 'pts'), ('observations', 'actual'), ('strong_tracks', 'strong')))


def previous_row(stats):
    """obs[obs.frame.eq(frame)].iloc[-1] for a track table that is only read.

    A numeric frame column equals frame exactly where its Python value does,
    and a dictionary finds those values (equal numbers hash alike; NaN matches
    nothing); the last position written is the filtered table's last row, and
    iloc of that position gives the same row (values, name, dtype). The row is
    only read. A frame absent from the table (IndexError) or another column
    type takes the original statement."""
    held = {}

    def _previous_row(obs, frame):
        entry = held.get(id(obs))
        if entry is None or entry[0] is not obs:
            column = obs['frame'] if 'frame' in obs.columns and obs.columns.is_unique else None
            positions = None
            if column is not None and column.dtype.kind in 'iuf':
                positions = {}
                for position, value in enumerate(column.tolist()):
                    positions[value] = position
            if len(held) >= 4096:
                held.clear()
            entry = held[id(obs)] = (obs, positions)
        position = entry[1].get(frame) if entry[1] is not None else None
        if position is None:
            stats['previous_original'] = stats.get('previous_original', 0) + 1
            return obs[obs.frame.eq(frame)].iloc[-1]
        stats['previous_indexed'] = stats.get('previous_indexed', 0) + 1
        return obs.iloc[position]
    return _previous_row


def held_nodes(stats):
    index = {}

    def _held_nodes(records, frame, owner):
        if index.get('records') is not records:
            table = {}
            for q in records[1:]:
                table.setdefault((q['frame'], q['identity']), set()).add(q['node'])
            index.update(records=records, table=table)
            stats['indexes'] += 1
        stats['lookups'] += 1
        return set(index['table'].get((frame, owner), ()))
    return _held_nodes


class PointRow:
    __slots__ = ('frame', 'y', 'x', 'radius_px')

    def __init__(self, frame, y, x, radius_px):
        self.frame = frame
        self.y = y
        self.x = x
        self.radius_px = radius_px


def point_rows(stats):
    import numpy as np
    names = ('frame', 'y', 'x', 'radius_px')

    def _point_rows(pts):
        columns = pts.columns
        if (not columns.is_unique or not all(name in columns for name in names)
                or not all(isinstance(pts[name]._values, np.ndarray) for name in names)):
            stats['iterrows'] = stats.get('iterrows', 0) + 1
            return pts.iterrows()
        stats['column_rows'] = stats.get('column_rows', 0) + 1
        return zip(pts.index, map(PointRow, *(pts[name].tolist() for name in names)))
    return _point_rows


def _repaired_source(function, module):
    """The runtime repair's function text when the repair has swapped in its code, else None.

    The core-replacement evidence repair compiles its patched function text
    under its own file name and swaps that code into the frozen function, so
    the frozen file no longer holds the running statements. The repair module's
    repaired_function rebuilds the same text from the frozen file; it is used
    only when it compiles to exactly the running code object.
    """
    import types
    code = function.__code__
    if code.co_filename == getattr(module, '__file__', None):
        return None
    repair = next((m for m in list(sys.modules.values())
                   if getattr(m, '__file__', None) == code.co_filename and hasattr(m, 'repaired_function')), None)
    if repair is None:
        return None
    _, patched = repair.repaired_function(Path(module.__file__).read_text(encoding='utf8'))
    compiled = compile(patched, code.co_filename, 'exec')
    rebuilt = next((c for c in compiled.co_consts
                    if isinstance(c, types.CodeType) and c.co_name == function.__name__), None)
    if rebuilt is None or rebuilt != code:
        raise ValueError('running code is not the repair of the verified version')
    return patched + '\n'


def _rewritten(function, module, stats):
    """The rewrite of the function's running statements, or None."""
    try:
        source = _repaired_source(function, module)
        if source is None:
            source = inspect.getsource(function)
    except (OSError, TypeError, ValueError, SyntaxError):
        return None
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            return None
        source = source.replace(old, new)
    if source.count(PREVIOUS[0]) == 1:
        source = source.replace(*PREVIOUS)
    if source.count(DISC[0]) == 1:
        source = source.replace(*DISC)
    for old, new in GROUPS:
        if source.count(old) == 1:
            source = source.replace(old, new)
    namespace = dict(vars(module))
    for key in ('groups_in_order', 'groups_sorted'):
        stats.setdefault(key, 0)
    namespace['_tracks_in_order'] = _sort_skips.tracks_in_order(stats)
    namespace['_held_nodes'] = held_nodes(stats)
    namespace['_point_rows'] = point_rows(stats)
    namespace['_previous_row'] = previous_row(stats)
    namespace['_nearest_node'] = nearest_node(stats)
    exec(compile(source, '<core-replacement-index>', 'exec'), namespace)
    stats['adapted'] += 1
    return namespace['preserve_core_replacement_owners']


def adapt(function, stats):
    """The verified frozen version, rewritten at call time from its running code; else the function itself.

    A runtime repair may swap the frozen function's code after this adapter is
    handed the function (it finds the function through __wrapped__), so each
    call rewrites the code running at that moment, kept while that code stays.
    """
    module = sys.modules.get(getattr(function, '__module__', None))
    if (getattr(function, '__name__', None) != 'preserve_core_replacement_owners' or module is None
            or getattr(module, 'preserve_core_replacement_owners', None) is not function):
        return function
    try:
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE:
            return function
    except (OSError, TypeError):
        return function
    held = {}

    def preserve_core_replacement_owners(*args, **kwargs):
        code = function.__code__
        if held.get('code') is not code:
            held['code'] = code
            held['function'] = _rewritten(function, module, stats) or function
        return held['function'](*args, **kwargs)
    preserve_core_replacement_owners.__wrapped__ = function
    return preserve_core_replacement_owners
