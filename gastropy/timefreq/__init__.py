"""
GastroPy Time-Frequency Module
================================

Per-band time-frequency decomposition of gastric signals.

This module provides functions for decomposing EGG signals across
gastric frequency bands (bradygastria, normogastria, tachygastria),
extracting narrowband phase, amplitude, cycle durations, and
summary metrics for each band.
"""

from .decompose import band_decompose, multiband_analysis

__all__ = [
    "band_decompose",
    "multiband_analysis",
]
