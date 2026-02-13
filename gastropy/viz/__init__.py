"""
GastroPy Visualization Module
===============================

Visualization utilities for EGG data and gastric-brain analyses.

Functions
---------
plot_psd
    Power spectral density with gastric band shading.
plot_egg_overview
    4-panel EGG overview (raw, filtered, phase, amplitude).
plot_cycle_histogram
    Cycle duration distribution with normogastric bounds.
plot_artifacts
    Phase time series with artifact regions highlighted.
plot_volume_phase
    Per-volume mean phase from fMRI-EGG coupling.
plot_egg_comprehensive
    Multi-panel comprehensive EGG analysis figure.
"""

from .plotting import (
    plot_artifacts,
    plot_cycle_histogram,
    plot_egg_comprehensive,
    plot_egg_overview,
    plot_psd,
    plot_volume_phase,
)

__all__ = [
    "plot_psd",
    "plot_egg_overview",
    "plot_cycle_histogram",
    "plot_artifacts",
    "plot_volume_phase",
    "plot_egg_comprehensive",
]
