"""Repeated saved values can share decoding without sharing mutable objects."""
import json

import pytest

from pymicroglia.pipelines._tables import _cell_decoder


def test_cell_decoder_preserves_json_types_and_nested_independence():
    decode = _cell_decoder()
    cells = ['null', 'true', 'false', '0', '-0.0', '1.25', '1e20',
             '"é"', ' " spaced " ', '[1,{"a":[2]}]', '{"a":[1]}', '"'+('x'*200)+'"']
    for cell in cells * 2:
        actual, expected = decode(cell), json.loads(cell)
        assert actual == expected
        assert type(actual) is type(expected)
    first = decode('{"a":[1]}')
    first['a'].append(2)
    assert decode('{"a":[1]}') == {'a': [1]}
    for i in range(5000):
        assert decode(str(i)) == i
    assert decode('true') is True


@pytest.mark.parametrize('cell', ['1,2', '[1', '{"a":}', '', '\ufeff1', 'null junk'])
def test_cell_decoder_preserves_invalid_input_errors(cell):
    with pytest.raises(json.JSONDecodeError):
        _cell_decoder()(cell)


def test_table_read_observes_edits_and_keeps_nested_cells_independent(tmp_path, monkeypatch):
    import csv
    from pymicroglia.pipelines import _tables
    path = tmp_path/'values.csv'
    contract = {'schema_version': 2, 'columns': ['label', 'value'],
                'dtypes': {'label': 'object', 'value': 'object'}}
    monkeypatch.setattr(_tables, 'schema', lambda _: contract)

    def write(value):
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(contract['columns'])
            writer.writerows([[json.dumps('cell'), json.dumps(value)]] * 2)

    write({'measurements': [1]})
    frame = _tables.read(path)
    frame.at[0, 'value']['measurements'].append(2)
    assert frame.at[1, 'value'] == {'measurements': [1]}
    write({'measurements': [3]})
    assert _tables.read(path).at[0, 'value'] == {'measurements': [3]}
