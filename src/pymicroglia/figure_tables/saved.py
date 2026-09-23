"""Read measured tables and the options that produced them, in either layout."""
from pathlib import Path
import pandas as pd
from .._results import read_document, document


class Tables:
    def __init__(self, run, stem=None):
        self.run = Path(run).resolve()
        self.manifest = read_document(self.run / "manifest.json")
        self.stem = stem
        movies = self.manifest.get("movies", [])
        if self.stem is None and len(movies) == 1:
            self.stem = movies[0]["stem"]
        self.sources = [document(self.run / "manifest.json")]

    def table(self, name, *, optional=False, scope='movie'):
        filename = name if name.endswith(".csv") else name + ".csv"
        paths = []
        if self.stem is not None and scope != 'run':
            for kind in ("measure", "tracker", "windows"):
                paths.append(self.run / kind / self.stem / filename)
            for kind in ("tables", "derived", "tracker", "windows"):
                paths.append(self.run / self.stem / kind / filename)
        paths.extend([self.run / "pooled" / filename, self.run / "pooled" / "tables" / filename,
                      self.run / filename])
        found = next((p for p in paths if p.is_file()), None)
        if found is None:
            if optional:
                return None
            raise FileNotFoundError(f"{filename} is unavailable in {self.run}")
        self.sources.append(found)
        try:
            frame = pd.read_csv(found)
        except pd.errors.EmptyDataError:
            if optional:
                return None
            raise
        if self.stem is not None and scope != 'run' and "stem" in frame:
            frame = frame.loc[frame.stem.eq(self.stem)].copy()
        return frame

    def module_params(self, name):
        matches = []
        for movie in self.manifest.get("movies", []):
            if self.stem is not None and movie.get("stem") != self.stem:
                continue
            matches.extend(row.get("parameters", {}) for row in movie.get("modules", [])
                           if row["module"] == name)
        if matches and any(row != matches[0] for row in matches[1:]):
            raise ValueError(f"Movies have different {name} settings; choose a stem")
        return matches[0] if matches else {}

    def stack(self,name):
        import tifffile
        paths=[self.run/'stacks'/str(self.stem)/name,self.run/str(self.stem)/'stacks'/name]
        path=next((p for p in paths if p.is_file()),None)
        if path is None:raise FileNotFoundError(f'{name} is unavailable for {self.stem}')
        self.sources.append(path)
        return tifffile.imread(path)

    @property
    def movie(self):
        movies = [m for m in self.manifest.get('movies',[]) if m.get('stem')==self.stem]
        if len(movies)!=1:
            raise ValueError('Choose stem to identify one movie')
        return movies[0]

    @property
    def summary(self):
        return self.movie['summary']

    @property
    def field(self):
        return self.movie['provenance']['field']

    @property
    def scale(self):
        from ..measure.context import Scale
        details = self.summary.get('scale',{})
        return Scale(self.interval,details.get('microns_per_pixel'),details.get('source','uncalibrated'))

    @property
    def interval(self):
        movies = [m for m in self.manifest.get('movies', [])
                  if self.stem is None or m.get('stem') == self.stem]
        intervals = {float(m['summary']['scale']['minutes_per_frame']) for m in movies}
        if len(intervals) != 1:
            raise ValueError('Movies have different frame intervals; choose a stem')
        return intervals.pop()

    def input_path(self,name):
        import hashlib
        row = self.movie.get('provenance',{}).get('inputs',{}).get(name)
        if row is None:
            return None
        path = Path(row['path'])
        if not path.is_absolute():
            path = self.run/path
        if row.get('sha256'):
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream,'sha256').hexdigest()
            if digest!=row['sha256']:
                raise ValueError(f'The recorded {name} input changed: {path}')
        self.sources.append(path)
        return path
