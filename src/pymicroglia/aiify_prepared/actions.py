"""Reviewed read-only shortcut; the host supplies its native current-state readback."""
_overview = None

def bind(app):
    global _overview
    _overview = app.state.aiify_overview

def workspace_overview():
    """Read the native workspace state without navigating or submitting work."""
    if _overview is None:
        raise RuntimeError("This shortcut must be bound to its running app.")
    return _overview()
