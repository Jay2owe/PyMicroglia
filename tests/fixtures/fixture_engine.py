#!/usr/bin/env python
"""A synthetic engine, shaped like a real one, for testing the reader."""

METHOD_VERSION = "2026-08-19-fixture"

# ============================ PROTOCOL PARAMETERS ============================
# Change values here, not the logic below.

DEFAULT_SERIES = 0                  # Which series to read.
DEFAULT_THRESHOLD_SIGMA = 12.0      # A pixel is a spike when it exceeds the
                                    # neighbour maximum by this many robust
                                    # temporal noise sigmas. CAUTION: lowering
                                    # this starts replacing real bright
                                    # transients.
DEFAULT_MASK_GROWTH_PX = 2          # Dilate each spike by this many px.
DEFAULT_WRITE_MASK = True           # Write the replaced-pixel mask.
DEFAULT_VIDEO_LUT = "#a340d1"       # A hex value, to prove the comment split
                                    # is quote-aware.
# The description for this one sits above it instead of beside it.
DEFAULT_UNDERLAY_GREY = "0.72"
DEFAULT_DISPLAY_PERCENTILES = (0.5, 99.5)   # Low and high display percentiles.
# ========================== END PROTOCOL PARAMETERS ==========================


def main():
    raise SystemExit("fixture: never executed by the test suite")
