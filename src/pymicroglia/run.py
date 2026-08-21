"""Calling an action by name, with the record around it."""

from __future__ import annotations

from typing import Any

from .recording import capture
from .knowledge import validate
from .registry import REGISTRY, ClaimRequired, require_claim

__all__ = ["run_action", "run_recorded", "ActionPending", "ActionInvalid",
           "ClaimRequired"]


class ActionPending(NotImplementedError):
    """The action is declared but its stage has not landed yet."""


class ActionInvalid(ValueError):
    """The action name or its arguments are wrong."""


def run_action(action: str, *, claim: str = "", output_roots=(),
               notebook: bool = False, request: str = "", entry: str = "",
               **params: Any) -> Any:
    """Run one registered action and record it, returning what it returned.

    What a caller at a prompt or in a notebook wants: the cleaned series, the
    trace table, the figure.
    """
    return run_recorded(action, claim=claim, output_roots=output_roots,
                        notebook=notebook, request=request, entry=entry,
                        **params)["result"]


def run_recorded(action: str, *, claim: str = "", output_roots=(),
                 notebook: bool = False, request: str = "", entry: str = "",
                 **params: Any) -> dict[str, Any]:
    """The same run, handing back the record as well as the result.

    The command line asks for this rather than for the return value alone: what
    a person needs after a run is the run id to look it up by and the list of
    files it wrote, not whatever the function happened to return — which for a
    cleaned stack is a boolean array with twenty-six million entries in it.

    Validation happens before anything is written, so a typo costs nothing —
    and that includes the claim. An action that concludes something is refused
    without one here rather than six hours later, or worse, recorded with a
    sentence generated from its own summary that nobody can search on.
    """
    check = validate(action, params)
    if check.get("error") == "unknown_action":
        raise ActionInvalid(
            f"{check['message']} Available: {', '.join(check['available'])}"
        )
    if check.get("unknown_params"):
        raise ActionInvalid(check["message"])
    require_claim(action, claim)

    function = REGISTRY.resolve(action)
    if function is None:
        raise ActionPending(
            f"{action} is not implemented yet: it binds to "
            f"{REGISTRY.binds_to(action)!r}, which does not exist. "
            "Its stage in docs/pymicroglia/ has not been run."
        )

    with capture(action, params, claim=claim, output_roots=output_roots,
                 notebook=notebook, request=request, entry=entry) as run:
        run.result = function(**params)
    return {"result": run.result, "record": dict(run.record),
            "recorded": bool(run.recorded),
            "notebook": getattr(run, "notebook", None)}
