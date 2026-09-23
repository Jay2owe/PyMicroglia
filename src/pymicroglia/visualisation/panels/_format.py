"""Convert saved display values into numbers and labels without fitting them."""
import math


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def numeric(column, **unused):
    return column.map(number).astype(float)


def missing(value):
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def present(value):
    return not missing(value)
