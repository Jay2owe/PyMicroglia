"""Declared contrast helpers, forwarded to the scientific owner."""
from pymicroglia import workbench
from .contrasts import ContrastSpec, parse_contrasts

def __getattr__(name):
    return getattr(workbench.group_contrasts, name)
