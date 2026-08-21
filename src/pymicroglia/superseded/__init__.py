"""Methods this package no longer runs, kept because records of them exist.

A run record stores the script that reproduces its run, and that script names
the function by its import path. Delete the function and every record written
before the replacement stops being a result and becomes a note about one. So a
superseded method moves here rather than out: nothing calls it, no action binds
to it, no catalogue entry mentions it, and ``filtering.remove_cosmic_rays``
still resolves for a record that asks for it.

Nothing new belongs in this package. Code arrives here on the day it stops
being how the analysis is done, carrying the version string it had, and does
not change afterwards — a superseded method that gets edited no longer
reproduces the runs that are the whole reason it was kept.
"""

from __future__ import annotations

__all__ = ["matched_line"]
