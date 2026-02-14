"""
GastroPy Neuro Module
======================

Neuroimaging utilities for gastric-brain coupling analyses.

This module provides preprocessing and utility functions for working
with fMRI, EEG, and MEG data in the context of gastric-brain coupling
research.

Submodules
----------
fmri — fMRI-specific utilities (scanner triggers, volume windowing)
eeg  — EEG preprocessing and utilities
meg  — MEG preprocessing and utilities
"""

from .fmri import (
    apply_volume_cuts,
    bold_voxelwise_phases,
    compute_plv_map,
    compute_surrogate_plv_map,
    create_volume_windows,
    find_scanner_triggers,
    phase_per_volume,
    regress_confounds,
)

__all__ = [
    "find_scanner_triggers",
    "create_volume_windows",
    "phase_per_volume",
    "apply_volume_cuts",
    "regress_confounds",
    "bold_voxelwise_phases",
    "compute_plv_map",
    "compute_surrogate_plv_map",
]
