"""
GastroPy Metrics Module
=========================

Metric extraction from EGG signals.

This module provides functions for computing gastric rhythm metrics
including band power, instability coefficient, cycle duration statistics,
and quality assessment.
"""

from .bands import (
    BRADYGASTRIA,
    GASTRIC_BANDS,
    NORMOGASTRIA,
    TACHYGASTRIA,
    GastricBand,
    band_power,
)
from .quality import assess_quality
from .stability import cycle_stats, instability_coefficient, proportion_normogastric

__all__ = [
    "GastricBand",
    "BRADYGASTRIA",
    "NORMOGASTRIA",
    "TACHYGASTRIA",
    "GASTRIC_BANDS",
    "band_power",
    "instability_coefficient",
    "cycle_stats",
    "proportion_normogastric",
    "assess_quality",
]
