"""One-time reproducible port of the approved extras markup (no demo data)."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
demo = (ROOT / "docs/workbench-layout/demo/index.html").read_text(encoding="utf-8")
html = demo[demo.index('        <div class="transport"'):demo.index('        <footer>')]
html = html.replace('          <label>Cell ', '          <label>Recording <select id="recording-select" aria-label="Recording"></select></label>\n          <label>Cell ')
html = html.replace('value="72"', 'value="0"')
html = html.replace('value="refined">Refined', 'value="accepted">Saved masks').replace('Base + refined', 'Base + saved')
html = html.replace('Refined <i', 'Saved <i')
html = html.replace('<label><input id="show-change" type="checkbox"> Mask changes</label>', '')
html = html.replace('Simulated phase', 'Saved phase').replace('Simulated group', 'Condition').replace('Simulated groups', 'Condition')
html = html.replace('Synthetic signal image', 'Saved source image').replace('Synthetic states, observation gaps and intervention time', 'Saved events and observation gaps')
html = html.replace('Simulated cell phases on a 24 hour clock', 'Saved peak phases as fractions of their fitted periods')
html = html.replace('Gold line = simulated intervention at 60 h.', 'Only saved events, windows and observed gaps are shown.')
html = html.replace('Positions use simulated phases, not fitted results.', 'Positions use saved fitted phases and their periods.')
html = html.replace('<label>Neighbour distance <input id="neighbour-distance" type="range" min="55" max="160" value="100"> <output id="distance-label">100 px</output></label>', '<label>Saved links <select id="link-source"></select></label>')
html = html.replace('Lines mean spatial proximity in this demo; they do not imply biological coordination.', 'Links use saved contact or relationship results. Their meaning is shown below the plot.')
html = html.replace('These simulated groups are not experimental replicates; no significance test is reported.', 'Cells are observations within their recording; this preview reports no significance test.')
html = re.sub(r'id="([^"]+)"', r'id="xp-\1"', html)
html = html.replace('href="#', 'href="#xp-')
html = '<div class="xp-root">\n' + html + '\n<p class="xp-provenance" id="xp-provenance"></p></div>\n'
(ROOT / "src/pymicroglia/ui/static/exploration.html").write_text(html, encoding="utf-8")
