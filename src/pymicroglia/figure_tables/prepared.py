"""Frozen values, evidence and drawing instructions produced before rendering."""
from dataclasses import dataclass, field


class PreparedViews(dict):
    """Named view inputs plus the evidence and wording shared by their page."""
    def __init__(self, views, *, auxiliary=None, wording=None):
        super().__init__(views)
        self.auxiliary = auxiliary or {}
        self.wording = wording or {}


@dataclass
class Drawing:
    renderer: object
    args: tuple
    kwargs: dict = field(default_factory=dict)

    def render(self, canvas):
        return self.renderer(*self.args, **self.kwargs, canvas=canvas)


@dataclass
class PreparedPage:
    drawing: Drawing
    figure_data: object
    auxiliary: dict = field(default_factory=dict)
    heading: str = ""
    readme: str = ""
    subtitle: str = ""
    footnote: str = ""
    title_fields: dict = field(default_factory=dict)
    producer_sources: dict = field(default_factory=dict)
    standalone_producer: str = ""
    wording: object = None
    views: dict = field(default_factory=dict)

    def for_view(self, name):
        if name is None or name == 'page' and not self.views:
            return self
        if name not in self.views:
            raise ValueError(f'Prepared view {name!r} is unavailable; choose {", ".join(self.views)}')
        from dataclasses import replace
        drawing,table=self.views[name]
        return replace(self,drawing=drawing,figure_data=table,views={})
