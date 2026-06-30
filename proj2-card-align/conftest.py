"""
conftest.py — pytest path setup for the card-alignment project.

Sits at the proj2 root. pytest runs the nearest conftest before collecting tests
beneath it; the line below puts this project's folder on sys.path so the test
files can import the 'card_*' modules by name.
"""

import os
import sys

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)
