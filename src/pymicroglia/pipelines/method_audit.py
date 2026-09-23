"""Run the method audit workflow on saved measurement tables."""
from .audit.options import AuditRequest as REQUEST
from .audit.workflow import RECIPE
STAGES = tuple(step.name for step in RECIPE.steps)
METHOD_VERSION = "1"


def method_audit(run, pipeline_request, *, output_dir=None, only=None, presentation=None,
          if_exists="version", run_label=None, claim=""):
    from ._requests import execute
    return execute("method_audit", run, pipeline_request, output_dir=output_dir, only=only,
                   presentation=presentation, if_exists=if_exists, run_label=run_label, claim=claim)


run = method_audit
