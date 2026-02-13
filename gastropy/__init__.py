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
egg        — Core EGG processing (import, clean, process, analyze)
signal     — General signal processing utilities
timefreq   — Time-frequency analysis (wavelets, spectrograms)
metrics    — Metric extraction (peak frequency, instability, etc.)
coupling   — Gastric-brain phase coupling analyses
neuro      — Neuroimaging utilities (fMRI, EEG, MEG)
viz        — Visualization
io         — Data I/O and BIDS support
stats      — Statistical utilities
data       — Sample datasets
misc       — Miscellaneous utilities
"""

__version__ = "0.1.0"

# Flat namespace re-exports — uncomment as modules gain content
# from .egg import *
# from .signal import *
# from .timefreq import *
# from .metrics import *
# from .coupling import *
# from .neuro import *
# from .viz import *
# from .io import *
# from .stats import *
# from .data import *
# from .misc import *
