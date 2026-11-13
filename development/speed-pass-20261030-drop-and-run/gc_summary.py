"""Sum Java collection pauses in heap_probe.py logs: python gc_summary.py HEAP_FOLDER"""
import re
import sys
from pathlib import Path

for folder in sorted(Path(sys.argv[1]).glob("*-*")):
    log = folder / "gc.log"
    if not log.exists():
        continue
    text = log.read_text(errors="replace")
    pauses = [(line, float(m.group(1))) for line in text.splitlines()
              if (m := re.search(r"([\d.]+)ms\s*$", line))]
    full = [ms for line, ms in pauses if "Pause Full" in line]
    stamps = re.findall(r"^\[([\d.]+)s\]", text, re.M)
    print(folder.name, "pauses", len(pauses), "pause_s", round(sum(ms for _, ms in pauses) / 1000, 1),
          "full", len(full), "full_s", round(sum(full) / 1000, 1), "log_span_s", stamps[-1] if stamps else None)
