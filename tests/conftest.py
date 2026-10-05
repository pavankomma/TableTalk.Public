"""Build the demo database once if it is missing, so a fresh clone can run
the suite with no setup beyond `pip install`. Tests derive their
expectations from this DB at runtime instead of hard-coding counts."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if "TABLETALK_DB" not in os.environ:
    demo = ROOT / "data" / "demo.db"
    if not demo.exists():
        from scripts.build_demo_db import build

        build(demo)
