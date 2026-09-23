"""Resolve recorded reference traces without repeating recording-level scalars."""
from dataclasses import replace

from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from pymicroglia.pipelines._contracts import Measurement, text_key
from pymicroglia.pipelines.rhythm.discovery import MeasurementChoice, _resolve_measurement


def declared_unit(measurement, units):
    unit = units.get(measurement.column)
    if unit is None: return measurement
    text_key(unit, 'shared_reference.units.'+measurement.column)
    if measurement.unit and unit != measurement.unit:
        raise ValueError(measurement.column+': declared reference unit conflicts with its recorded unit')
    return replace(measurement, unit=unit)


def resolve_external(column, tables, declarations, grains, units):
    candidates = []; problems = []
    for name, frame in tables.items():
        if column not in frame: continue
        grain = grains.get(name)
        if grain is None:
            problems.append(name+': reference table grain is undeclared'); continue
        if 'identity' in grain:
            try: candidates.append(_resolve_measurement(MeasurementChoice(column, table=name), tables, declarations, grains, testing=True))
            except ValueError as error: problems.append(str(error))
            continue
        if set(grain)-{'stem'} not in [{'frame_index'}, {'hours'}] or 'identity' in frame:
            problems.append(name+': external reference must be an original recording trace or a per-cell trace'); continue
        required = list(dict.fromkeys(['stem', *grain, 'hours']))
        if not frame.columns.is_unique or set(required)-set(frame):
            problems.append(name+': reference has missing or duplicated clock columns'); continue
        keys = list(dict.fromkeys(['stem', *grain]))
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            problems.append(name+': missing or duplicate reference observation keys'); continue
        if column in set(required):
            problems.append(name+': a reference must be a measured outcome, not an identity or clock'); continue
        invalid = False
        for field in [column, 'hours']:
            values = frame[field]
            if is_bool_dtype(values.dtype) or is_complex_dtype(values.dtype) or (not is_numeric_dtype(values.dtype) and not values.isna().all()):
                problems.append(name+': original reference values and hours must be real numeric columns'); invalid = True
        if invalid: continue
        for movie in frame.stem.unique(): text_key(movie, 'external reference recording')
        declaration = declarations.get(column)
        candidates.append(Measurement(column, name, tuple(grain), declaration.label if declaration else column,
            declaration.unit if declaration else '', declaration is not None))
    if len(candidates) != 1:
        raise ValueError(column+(': ambiguous external reference sources' if candidates else ': unsuitable external reference; '+'; '.join(problems)))
    return declared_unit(candidates[0], units)
