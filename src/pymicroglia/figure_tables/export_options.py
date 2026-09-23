"""The shared figure export parameters, taken from the established save action."""
import inspect
from ..visualisation.save_actions import save_for


_parameters=inspect.signature(save_for).parameters
_names=list(_parameters)
NAMES=tuple(name for name in _names[_names.index('figure_profile'):_names.index('suffix')]
            if name != 'statistical_specs')  # Computed evidence belongs to preparation.
DEFAULTS={name:_parameters[name].default for name in NAMES}


def take(options):
    selected={key:options.pop(key) for key in NAMES if key in options}
    for name in ('width','height'):
        source='render_'+name+'_in'
        if source in selected:
            selected[name]=selected.pop(source)
    return selected
