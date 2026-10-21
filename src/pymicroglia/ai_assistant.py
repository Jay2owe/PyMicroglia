"""Optional shared assistant; every analysis goes through the UI's real job manager."""
from pathlib import Path
from . import __version__


def overview(app):
    rows = app.state.jobs.list()
    keys = ("id", "action", "state", "submitted_at", "error")
    return {"version": __version__, "jobs": [{k: row[k] for k in keys if k in row} for row in rows[:10]],
            "total_jobs": len(rows),
            "auto_url": app.state.auto_url, "motion_url": app.state.motion_url}


def current_state(app):
    return {"version": __version__, "active_jobs": sum(row.get("state") in ("queued", "running")
                                                        for row in app.state.jobs.list())}


class JobsSource:
    def __init__(self, manager):
        self.manager = manager

    def names(self):
        from .registry import REGISTRY
        return REGISTRY.names()

    def describe(self, name=None):
        if name is None:
            from .catalogue import action
            return {"actions": [{"name": key, "summary": (action(key) or {}).get("summary", ""),
                     "mutates": True, "destructive": bool((action(key) or {}).get("destructive"))}
                    for key in self.names()]}
        from .knowledge import describe
        row = describe(name)
        if not row.get("ok"):
            return row
        return {**row, "mutates": True, "params": [*row["params"],
            {"name": "claim", "required": row.get("claim_required", False),
             "description": "What this analysis should show."},
            {"name": "request", "required": False, "description": "Original user request."}]}

    def run(self, name, params, *, confirm=False):
        values = dict(params)
        claim, request = values.pop("claim", ""), values.pop("request", "")
        try:
            return {"ok": True, "result": self.manager.submit(name, values, claim, request)}
        except (ValueError, TypeError, KeyError) as exc:
            return {"ok": False, "code": "failed", "error": str(exc)}


def install(app, *, options=None, engine_argv=None):
    from aiify import Agent
    from aiify.integration import helper_options, protect_local_assistant
    from . import context
    app.state.aiify_overview = lambda: overview(app)
    agent = Agent("pymicroglia", actions=JobsSource(app.state.jobs), guide=context,
        state=lambda: current_state(app), engine_argv=engine_argv, prewarm=False,
        limit_check_every=None,
        instructions="Use the existing actions and claim requirements; do not change scientific defaults. "
        "Analysis actions submit recorded jobs visible in this workbench. Inspect jobs before resubmitting. "
        "Use the on-screen controls for the current comparison, recordings and display choices.",
        **helper_options(Path(__file__).parent, env="PYMICROGLIA_AIIFY_OPTIONS",
                         routes=("/api/*",), overrides=options))
    agent.mount(app, inject=True, panel={"title": "PyMicroglia assistant", "layout": "float"})
    protect_local_assistant(app)
    app.state.agent = agent
    return agent
