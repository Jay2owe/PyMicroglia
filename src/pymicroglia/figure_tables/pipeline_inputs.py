"""Verified saved tables and settings for preparation of one pipeline figure."""
from pathlib import Path
import json


class SavedInputs:
    def __init__(self, *, run, spec, item, options, binding, cached=None, text=None):
        self.run, self.spec, self.item = Path(run), spec, item
        self.options, self.binding = options, binding
        self.cached = cached or {}
        self.argv = []
        self.text = text
        self.sources = {}
        self.saved = self._read_sources()

    def _read_sources(self):
        from ..pipelines._runner import SavedResult, _validate
        from ..pipelines._contracts import content_id, result_from_dict
        sources = {}
        for name, row in self.binding["results"].items():
            path = Path(row["ledger"])
            from .._results import read_document
            data = read_document(path)
            receipt = data["completions"][row["entry"]] if row["entry"] is not None else data
            if row['entry'] is not None:
                from ..pipelines._records import expand_completion
                from auto_organotypic.layout import owning_folder
                receipt=expand_completion(owning_folder(path),receipt)
            if path.name == "result.json" and "result" not in receipt:
                receipt = {"result":receipt,"result_sha256":content_id(receipt)}
            if content_id(receipt) != row["sha256"]:
                raise ValueError("Saved figure source was changed")
            encoded = receipt.get("result", receipt)
            if "result_sha256" in receipt and content_id(encoded) != receipt["result_sha256"]:
                raise ValueError("Saved figure result was changed")
            saved = SavedResult(Path(row["root"]), result_from_dict(encoded))
            _validate(saved)
            sources[name] = saved
        return sources

    def option(self, name):
        return self.options.get(name)

    def cache(self, name):
        return self.cached.get(name)

    def table(self, alias):
        from ..pipelines._screening import read_table
        step, artifact = self.binding["inputs"][alias]
        path = self.saved[step].artifact(artifact)
        self.record_source(alias, path)
        return read_table(path)

    def pipeline_metadata(self, step):
        saved = self.saved[step]
        names = {ref.name for ref in saved.outcome.artifacts}
        if "provenance" in names:
            path = saved.artifact("provenance")
            self.record_source(step+"_provenance", path)
            return json.loads(path.read_text(encoding="utf-8"))
        return saved.outcome.provenance.as_dict()

    def pipeline_selection(self,alias):
        step,name=self.binding['selections'][alias]
        matches=[row for row in self.saved[step].outcome.selections if row.name==name]
        if len(matches)!=1:raise ValueError('No unique saved selection '+alias)
        return matches[0]

    def has(self,name):
        alias,_,column=name.partition(':')
        if alias not in self.binding['inputs']:return False
        table=self.table(alias)
        return not column or column in table

    def record_source(self, name, path):
        self.sources[name] = Path(path)

    def _pipeline_sources(self):
        return self.binding, self.saved

    @property
    def name(self):
        return self.item or self.spec.key

    def pipeline_table_path(self, alias):
        step, artifact = self.binding["inputs"][alias]
        path = self.saved[step].artifact(artifact)
        self.record_source(alias,path)
        return path
