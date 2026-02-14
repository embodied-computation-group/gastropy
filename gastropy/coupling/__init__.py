"""
GastroPy Coupling Module
==========================

Gastric-brain phase coupling analyses.

This module provides modality-agnostic functions for computing
phase-locking values, surrogate testing, and circular statistics
that quantify the coupling between gastric rhythm and neural signals.

Functions
---------
phase_locking_value
    Compute PLV between two phase time series.
phase_locking_value_complex
    Compute complex PLV (magnitude + preferred phase lag).
surrogate_plv
    Compute surrogate PLV via circular time-shifting.
coupling_zscore
    Z-scored coupling strength (empirical vs. surrogate).
circular_mean
    Circular mean direction of phase values.
resultant_length
    Mean resultant length (phase consistency).
rayleigh_test
    Rayleigh test for non-uniformity of circular data.
"""

from .circular import circular_mean, rayleigh_test, resultant_length
from .plv import phase_locking_value, phase_locking_value_complex
from .surrogate import coupling_zscore, surrogate_plv

__all__ = [
    "phase_locking_value",
    "phase_locking_value_complex",
    "surrogate_plv",
    "coupling_zscore",
    "circular_mean",
    "resultant_length",
    "rayleigh_test",
]
