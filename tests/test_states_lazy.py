"""Reading settings and describing actions needs no learning runtime."""
import subprocess
import sys


def test_option_parsing_does_not_import_learning_libraries():
    code = '''
import sys
from pymicroglia.states.options import StateOptions
from pymicroglia.clustering.options import ClusteringOptions
StateOptions(); ClusteringOptions()
for name in ('sklearn', 'torch', 'hmmlearn', 'hdbscan'):
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_learning_actions_require_claims():
    import pytest
    from pymicroglia.states import states
    from pymicroglia.clustering import cluster
    from pymicroglia.registry import ClaimRequired
    for action in (states, cluster):
        with pytest.raises(ClaimRequired):
            action("unused")
