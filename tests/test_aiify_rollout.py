"""Boundary and live-backend checks for the shared assistant rollout."""
from pymicroglia.ai_assistant import JobsSource
from pymicroglia.ui.jobs import JobManager

def test_unknown_action_cannot_submit_a_job():
    manager = JobManager()
    try:
        source = JobsSource(manager)
        reply = source.run('missing-action', {})
        assert not reply['ok'] and not manager.list()
        assert 'measure' in source.names()
        contract = source.describe('measure')
        assert any(p['name']=='claim' for p in contract['params'])
        assert contract['mutates']
    finally:
        manager.pool.shutdown(wait=True)
