"""Run the rhythm discovery workflow on saved measurement tables."""
from .rhythm.discovery import RhythmDiscoveryRequest as REQUEST
from .rhythm.discovery import RECIPE
STAGES = tuple(step.name for step in RECIPE.steps)
METHOD_VERSION = "1"


def rhythm_discovery(run, pipeline_request, *, output_dir=None, only=None, presentation=None,
          if_exists="version", run_label=None, claim=""):
    from ._requests import execute
    return execute("rhythm_discovery", run, pipeline_request, output_dir=output_dir, only=only,
                   presentation=presentation, if_exists=if_exists, run_label=run_label, claim=claim)


run = rhythm_discovery
