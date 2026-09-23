"""Run the intervention response workflow on saved measurement tables."""
from .intervention.options import InterventionRequest as REQUEST
from .intervention.options import RECIPE
STAGES = tuple(step.name for step in RECIPE.steps)
METHOD_VERSION = "1"


def intervention_response(run, pipeline_request, *, output_dir=None, only=None, presentation=None,
          if_exists="version", run_label=None, claim=""):
    from ._requests import execute
    return execute("intervention_response", run, pipeline_request, output_dir=output_dir, only=only,
                   presentation=presentation, if_exists=if_exists, run_label=run_label, claim=claim)


run = intervention_response
