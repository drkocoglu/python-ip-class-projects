"""Project 4 - the executable entry point. This is the ONLY file you run:

    python scripts/proj4_main.py

It adds ``src/`` to the import path, then uses the reusable modules in
``src/proj4_ip/`` to:
  1. find and load the input image (data lives OUTSIDE this project),
  2. extract the periodic pattern (band-pass + spike threshold),
  3. correct the non-uniform illumination (high-pass),
  4. save the two results and all figures into a ``results/`` folder that is
     created here at run time.

The reusable modules in src/ are libraries - they are imported by this script,
not run on their own. Tunable filter numbers live in src/proj4_ip/config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

PATTERN_OUTPUT = "Proj4_pattern.tif"
UNIFORM_OUTPUT = "Proj4_uniform.tif"


def _add_src_to_path() -> None:
    """Make the reusable package importable (no install / no pyproject needed)."""
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    """Run the full Project 4 pipeline end to end."""
    _add_src_to_path()

    from proj4_ip import dataio as io
    from proj4_ip.pipeline import correct_illumination, extract_pattern

    input_path = io.find_input_image()
    out_dir = io.results_dir(create=True)  # creates proj4/results/
    print(f"Input image : {input_path}")
    print(f"Results dir : {out_dir}")

    image = io.load_image(input_path)

    print("Task 1: extracting periodic pattern (band-pass + spike threshold)...")
    pattern_result = extract_pattern(image)
    print(f"  kept {pattern_result.kept_count} spectral spikes")

    print("Task 2: correcting non-uniform illumination (high-pass)...")
    illum_result = correct_illumination(image)

    # Save the two deliverable images first (needs no matplotlib).
    pattern_path = io.save_image(pattern_result.pattern, out_dir / PATTERN_OUTPUT)
    uniform_path = io.save_image(illum_result.corrected, out_dir / UNIFORM_OUTPUT)
    print(f"Saved: {pattern_path.name}")
    print(f"Saved: {uniform_path.name}")

    # Then the insightful figures; plotting is optional.
    try:
        from proj4_ip import visualize as viz
    except ImportError as exc:
        print(f"Plotting unavailable ({exc}); deliverable images were still saved.")
        return

    figures = viz.build_figures(image, pattern_result, illum_result)
    for path in viz.save_figures(figures, out_dir):
        print(f"Saved: {path.name}")

    print(f"\nDone. All images are in: {out_dir}")


if __name__ == "__main__":
    main()
