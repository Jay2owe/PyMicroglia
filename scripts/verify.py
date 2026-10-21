"""Project-local developer checks; independent of agent skill installation."""
from pathlib import Path
from project_contract import main

if __name__ == "__main__":
    raise SystemExit(main(default_root=Path(__file__).resolve().parents[1]))
