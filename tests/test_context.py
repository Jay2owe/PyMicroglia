"""Portable usage guidance remains available without scientific imports."""
import ast
import json
from pathlib import Path
import pytest
from pymicroglia import context


def test_every_topic_is_nonempty_and_searchable():
    for topic in context.topics():
        row=context.read(topic,format='json')
        assert row['topic']==topic and row['content'].startswith('# ')
    assert any(row['topic']=='rhythms' for row in context.search('period search range'))
    with pytest.raises(ValueError):context.read('../secrets')


def test_reader_imports_only_standard_library():
    tree=ast.parse(Path(context.__file__).read_text(encoding='utf-8'))
    imports={node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
    imports.update(alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names)
    assert imports<={'importlib','re'}


def test_packaged_bundle_matches_live_topics():
    bundle=json.loads(Path(context.__file__).with_name('pymicroglia_context.json').read_text(encoding='utf-8'))
    assert bundle['schema_version']==1
    assert bundle['topics']==[context.read(topic,format='json') for topic in context.topics()]
