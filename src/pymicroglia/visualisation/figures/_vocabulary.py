"""Shared meanings of figure options, independent of per-figure defaults."""
import json
from importlib.resources import files

OPTIONS = json.loads(files('pymicroglia.data').joinpath('figure_options.json').read_text(encoding='utf-8'))


def meaning(name):
    try:
        return OPTIONS[name]
    except KeyError:
        raise KeyError(f'Undeclared figure option {name!r}; add its shared meaning to figure_options.json') from None
