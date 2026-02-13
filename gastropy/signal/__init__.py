"""
GastroPy Signal Module
=======================

General-purpose signal processing utilities.

This module provides functions for spectral analysis, filtering,
phase extraction, cycle detection, and resampling.
"""

from .artifacts import detect_phase_artifacts, find_cycle_edges
from .filtering import apply_bandpass, design_fir_bandpass
from .phase import cycle_durations, instantaneous_phase, mean_phase_per_window
from .resampling import resample_signal
from .spectral import psd_welch

__all__ = [
    "psd_welch",
    "design_fir_bandpass",
    "apply_bandpass",
    "instantaneous_phase",
    "cycle_durations",
    "mean_phase_per_window",
    "resample_signal",
    "detect_phase_artifacts",
    "find_cycle_edges",
]
