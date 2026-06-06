"""Root conftest.py — ensure scripts/ is importable in tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add scripts/ to sys.path so tests can import fetch_corpus directly.
_scripts_dir = str(Path(__file__).parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
