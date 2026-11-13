"""Open the manual track editor, choosing the bundle or session folder if none is given."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manual_editing_ui import launch_manual_editor  # noqa: E402

EDITING = Path(__file__).resolve().parents[1] / "manual_editing"


def main() -> int:
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
    else:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(
            title="Choose an editing bundle or session folder",
            initialdir=str(EDITING), mustexist=True)
        root.destroy()
        if not chosen:
            return 0
        source = Path(chosen)
    launch_manual_editor(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
