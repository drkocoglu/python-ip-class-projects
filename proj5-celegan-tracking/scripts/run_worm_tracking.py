"""Run the worm-tracking pipeline end to end.

Just run this file (no arguments): it finds the single video under
``data/proj5_data/``, tracks the worm, and writes the annotated result into
``proj5-worm-tracking/results/``.

Tune behaviour in ``src/proj5_ip/tracking_config.py``.
"""

import sys
from pathlib import Path


def main() -> None:
    """Add ``src`` to the path, then run the pipeline."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from proj5_ip import run

    run()


if __name__ == "__main__":
    main()
