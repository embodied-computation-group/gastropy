"""
GastroPy EGG Module
====================

Core electrogastrography (EGG) signal processing pipeline.

This module provides functions for cleaning, processing, and
analyzing EGG data, including channel selection, bandpass filtering,
phase extraction, and metric computation.
"""

from .egg_process import egg_clean, egg_process
from .egg_select import select_best_channel, select_peak_frequency

__all__ = [
    "egg_process",
    "egg_clean",
    "select_best_channel",
    "select_peak_frequency",
]
