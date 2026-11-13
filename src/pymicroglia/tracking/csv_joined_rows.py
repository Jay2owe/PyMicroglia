"""Large CSV bodies written as joined lines instead of one csv.writer row at a time.

DataFrame.to_csv formats each chunk of rows with _get_values_for_csv and then
hands every field to csv.writer through write_csv_rows. For a table without
its index and with the default dialect (comma, double quote, minimal quoting,
doubled quotes, no escape character) the writer turns each field into text
exactly as csv does it (the string itself; '' for None; repr for a float;
str otherwise), quotes a field holding a comma, quote, carriage return or line
feed by wrapping it in quotes and doubling its quotes, joins the fields with
commas and ends each line with the dialect's terminator. The same chunk values
are formatted the same way here and written to the same handle as one string.
A table under MINIMUM_ROWS rows, an index, another dialect, a single column,
a field of another type or non-ASCII text keeps the original row writer.
"""
from contextlib import contextmanager
import csv
import hashlib
import inspect
import io
import types

import numpy as np
import pandas
from pandas._libs import writers as libwriters
from pandas.io.formats import csvs

MINIMUM_ROWS = 20000
SOURCES = {'_save_chunk': 'a14161cac739c8292d4b169f94592c8cf1fdfc5d2624ad0771cdb0b2306cf4d4',
           '_save_body': '5ec28fd7e6cfd7367b72171dbebd3d172509ae16202ff59154a2bc2cf80891dd',
           'save': '7ee7a3421ab5cf94b661b220c4b51887236458f6856e4dc78a1e388912426e64'}
TEXT_TYPES = frozenset((str, int, bool, float, type(None), tuple, list))


class _Writer:
    """csv.writer that also keeps its handle; every call goes to the real writer."""

    def __init__(self, handle, **options):
        self.real = csv.writer(handle, **options)
        self.handle = handle
        dialect = self.real.dialect
        self.plain = (isinstance(handle, io.TextIOBase) and dialect.delimiter == ','
                      and dialect.quotechar == '"' and dialect.quoting == csv.QUOTE_MINIMAL
                      and dialect.doublequote and dialect.escapechar is None
                      and dialect.lineterminator in ('\n', '\r\n'))

    @property
    def dialect(self):
        return self.real.dialect

    def writerow(self, row):
        return self.real.writerow(row)

    def writerows(self, rows):
        return self.real.writerows(rows)


def _fields(values):
    """Each value as csv.writer writes it, or None for a type left to csv."""
    kinds = set(map(type, values))
    if not kinds <= TEXT_TYPES:
        return None
    if kinds <= {str}:
        return values
    if float not in kinds and type(None) not in kinds:
        return list(map(str, values))
    return ['' if value is None else repr(value) if type(value) is float else str(value)
            for value in values]


def _quoted(fields):
    joined = ''.join(fields)
    if not (',' in joined or '"' in joined or '\r' in joined or '\n' in joined):
        return fields
    return [('"' + field.replace('"', '""') + '"')
            if (',' in field or '"' in field or '\r' in field or '\n' in field) else field
            for field in fields]


def _supported():
    if pandas.__version__ != '2.2.2':
        return False
    try:
        return all(hashlib.sha256(inspect.getsource(getattr(csvs.CSVFormatter, name)).encode()).hexdigest()
                   == digest for name, digest in SOURCES.items())
    except (OSError, TypeError):
        return False


@contextmanager
def install():
    stats = dict(supported=False, chunks=0, rows=0, original_chunks=0)
    if not _supported():
        yield stats
        return
    original_chunk = csvs.CSVFormatter._save_chunk
    original_csv = csvs.csvlib
    shim = types.ModuleType('csv')
    shim.__dict__.update(vars(csv))
    shim.writer = _Writer

    def _save_chunk(self, start_i, end_i):
        writer = self.writer
        if not (type(writer) is _Writer and writer.plain and self.nlevels == 0
                and len(self.data_index) >= MINIMUM_ROWS and len(self.cols) >= 2):
            stats['original_chunks'] += 1
            return original_chunk(self, start_i, end_i)
        slicer = slice(start_i, end_i)
        df = self.obj.iloc[slicer]
        res = df._get_values_for_csv(**self._number_format)
        data = list(res._iter_column_arrays())
        columns = None
        if len(data) == len(self.cols) and all(
                type(array) is np.ndarray and array.dtype == object and array.ndim == 1
                and len(array) == end_i - start_i for array in data):
            columns = []
            for array in data:
                fields = _fields(array.tolist())
                if fields is None:
                    columns = None
                    break
                columns.append(_quoted(fields))
        text = None
        if columns is not None:
            terminator = writer.dialect.lineterminator
            text = terminator.join(map(','.join, zip(*columns))) + terminator
            if not text.isascii():
                text = None
        if text is None:
            # The original call, with the values already made.
            stats['original_chunks'] += 1
            ix = self.data_index[slicer]._get_values_for_csv(**self._number_format)
            libwriters.write_csv_rows(data, ix, self.nlevels, self.cols, writer.real)
            return
        writer.handle.write(text)
        stats['chunks'] += 1
        stats['rows'] += end_i - start_i

    _save_chunk.__wrapped__ = original_chunk
    csvs.CSVFormatter._save_chunk = _save_chunk
    csvs.csvlib = shim
    stats['supported'] = True
    try:
        yield stats
    finally:
        csvs.CSVFormatter._save_chunk = original_chunk
        csvs.csvlib = original_csv
