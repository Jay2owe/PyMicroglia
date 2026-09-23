"""Descriptions expose each parser's accepted top-level request fields."""
import importlib
import pytest
from pymicroglia.pipelines._requests import FAMILIES
from pymicroglia.pipelines.request_contracts import contract

@pytest.mark.parametrize('action',FAMILIES)
def test_all_accepted_request_keys_are_described(action,monkeypatch):
 _,module_name,class_name,_=FAMILIES[action]
 module=importlib.import_module('pymicroglia.pipelines.'+module_name)
 observed=set()
 class Captured(Exception):pass
 def capture(block,keys,where):
  observed.update(keys)
  raise Captured
 monkeypatch.setattr(module,'_known_keys',capture)
 with pytest.raises(Captured):
  getattr(module,class_name).from_dict({'pipeline':FAMILIES[action][0]}, {})
 assert observed==set(contract(action))

def test_describe_exposes_nested_scientific_choices_and_disabled_rhythm_default():
 from pymicroglia.knowledge import describe
 row=next(p for p in describe('intervention_response')['params'] if p['name']=='pipeline_request')
 fields=row['properties']
 assert fields['rhythms']['default']=={'enabled':False}
 options=fields['rhythms']['properties']['analysis_options']['properties']
 assert options['period_min_hours']['default']==2
 assert options['period_max_hours']['default']==48
 assert options['min_observations']['default']==24
 assert options['min_cycles']['default']==3
 assert 'cosinor' not in options['significance_method']['choices']
 assert all(field['description'] for field in fields.values())


@pytest.mark.parametrize("action",["measure","contrasts"])
def test_contrast_description_requires_an_explicit_replication_unit(action):
 from pymicroglia.knowledge import describe
 row=next(p for p in describe(action)["params"] if p["name"]=="contrasts")
 unit=row["items"]["properties"]["unit"]
 assert unit["required"] is True and "default" not in unit
 assert set(unit["choices"])=={"cell","movie","subject"}
