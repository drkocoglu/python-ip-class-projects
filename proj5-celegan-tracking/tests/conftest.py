"""Make ``proj5_ip`` importable during tests without installing the package.

Because the project has no per-project ``pyproject.toml`` or virtual
environment, this conftest inserts the project's ``src`` directory onto
``sys.path`` so ``pytest`` works whether it is launched from the project
folder or from the monorepo root.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
