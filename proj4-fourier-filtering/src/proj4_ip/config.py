"""Project 4 - FILTER SETTINGS (edit these numbers to retune the filters).

This is the one place to change how the filters behave. Edit a value and re-run
``python scripts/proj4_main.py``. Each knob has a note on what it does and a
range to try.
"""

# -----------------------------------------------------------------------------
# TASK 1 - PERIODIC PATTERN
# Extracted with a BAND-PASS (keeps only a ring of frequencies) plus a spike
# threshold (keeps only the strong peaks inside that ring).
# -----------------------------------------------------------------------------

# Inner cut-off (high-pass side): frequencies BELOW this are removed. This is
# what strips the smooth illumination out of the pattern.
#   bigger -> removes more central "glow";  too big -> eats the pattern
#   try: 5 - 10
PATTERN_INNER_CUTOFF = 6.0

# Outer cut-off (low-pass side): frequencies ABOVE this are removed. This drops
# the high-frequency speckle / noise.
#   too large -> lets noise back in and the pattern looks grainy
#   try: 30 - 60
PATTERN_OUTER_CUTOFF = 40.0

# Roll-off steepness of the band-pass (higher = sharper edges).
PATTERN_ORDER = 2

# Spike threshold on log(1 + |spectrum|): only components stronger than this are
# kept. The main "cleanness" knob.
#   higher -> fewer spikes -> cleaner, smoother pattern
#   try: 7 - 12
PATTERN_LOG_THRESHOLD = 11.0

# Weak components are multiplied by this factor.
#   0.01 matches the original MATLAB; use 0.0 to remove them completely.
PATTERN_ATTENUATION = 0.01


# -----------------------------------------------------------------------------
# TASK 2 - NON-UNIFORM ILLUMINATION
# Corrected with a HIGH-PASS: removes the slow brightness gradient (the
# lighting) and keeps the texture detail.
# -----------------------------------------------------------------------------

# Cut-off: frequencies below this (the illumination) are removed.
#   bigger -> flattens the lighting harder;  try: 3 - 10
ILLUMINATION_CUTOFF = 5.0

# Roll-off steepness of the high-pass.
ILLUMINATION_ORDER = 2
