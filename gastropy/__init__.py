"""
GastroPy
========

A Python package for electrogastrography (EGG) signal processing
and gastric-brain coupling analysis.

GastroPy provides a complete scientific pipeline including data import,
signal cleaning, signal processing, signal extraction, time-frequency
analysis, metric extraction, and utilities for gastric-brain coupling
analyses with fMRI, EEG, and MEG data.

Submodules
----------
egg        — Core EGG processing (clean, process, analyze)
signal     — General signal processing utilities
metrics    — Metric extraction (band power, instability, etc.)
neuro      — Neuroimaging utilities (fMRI, EEG, MEG)
timefreq   — Per-band time-frequency decomposition
coupling   — Gastric-brain phase coupling analyses
viz        — Visualization
io         — Data I/O and BIDS support
stats      — Statistical utilities
data       — Sample datasets
misc       — Miscellaneous utilities
"""

__version__ = "0.1.0"

# Flat namespace re-exports for implemented modules
from .coupling import *  # noqa: F401, F403
from .data import *  # noqa: F401, F403
from .egg import *  # noqa: F401, F403
from .io import *  # noqa: F401, F403
from .metrics import *  # noqa: F401, F403
from .signal import *  # noqa: F401, F403
from .timefreq import *  # noqa: F401, F403
from .viz import *  # noqa: F401, F403

# Modules with content but not re-exported at top level
# (access via gastropy.neuro.fmri, etc.)
# from .neuro import *

# Modules not yet implemented — uncomment as they gain content
# from .stats import *
# from .misc import *
