"""The same complete fit controls on every figure that performs a fresh fit."""
from ._declare import Option
from ...workbench import CIRCADIAN_ANALYSIS_OPTION_DEFAULTS,PERIOD_METHODS,SIGNIFICANCE_METHODS


def circadian_options():
    return tuple(Option(name,value,choices=tuple(PERIOD_METHODS) if name=='fit_method' else SIGNIFICANCE_METHODS if name=='significance_method' else ())
                 for name,value in CIRCADIAN_ANALYSIS_OPTION_DEFAULTS.items())


def underlay_options():
    """Real channel images beneath a spatial map; off unless channels are named."""
    return (Option('underlay',[]),Option('underlay_luts',[]),Option('underlay_weights',[]),Option('underlay_alpha',.6))
