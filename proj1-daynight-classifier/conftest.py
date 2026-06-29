"""Pytest configuration: make proj1's source modules importable from the tests.

The ``daynight_*`` modules live one directory up in ``proj1/``. Adding that
directory to ``sys.path`` lets the tests here import them with a flat
``import daynight_x`` no matter where pytest is launched from -- the tests
folder, the project folder, or the monorepo root. The unique ``daynight_*``
names keep this collision-free next to other projects that do the same.
"""

import sys
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))
