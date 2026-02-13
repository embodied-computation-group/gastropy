"""EGG data quality assessment."""

import numpy as np


def assess_quality(n_cycles, cycle_sd, prop_normo):
    """Assess EGG recording quality based on standard criteria.

    Quality control thresholds are based on EGG literature conventions:
    sufficient cycles (>=10), stable rhythm (SD < 6s), and normogastric
    dominance (>=70% of cycles in 2-4 cpm range).

    Parameters
    ----------
    n_cycles : int
        Number of detected gastric cycles.
    cycle_sd : float
        Standard deviation of cycle durations in seconds.
    prop_normo : float
        Proportion of cycles in the normogastric range (0 to 1).

    Returns
    -------
    dict
        Dictionary with boolean quality flags:

        - ``sufficient_cycles`` : At least 10 cycles detected.
        - ``stable_rhythm`` : Cycle duration SD < 6 seconds.
        - ``normogastric_dominant`` : >= 70% normogastric cycles.
        - ``overall`` : Sufficient cycles AND (stable OR normogastric).

    Examples
    --------
    >>> from gastropy.metrics import assess_quality
    >>> qc = assess_quality(n_cycles=15, cycle_sd=3.2, prop_normo=0.8)
    >>> qc["overall"]
    True
    """
    sufficient_cycles = int(n_cycles) >= 10
    stable_rhythm = float(cycle_sd) < 6.0 if not np.isnan(cycle_sd) else False
    normogastric_dominant = float(prop_normo) >= 0.70 if not np.isnan(prop_normo) else False
    overall = sufficient_cycles and (stable_rhythm or normogastric_dominant)

    return {
        "sufficient_cycles": sufficient_cycles,
        "stable_rhythm": stable_rhythm,
        "normogastric_dominant": normogastric_dominant,
        "overall": overall,
    }
