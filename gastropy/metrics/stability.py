"""Gastric rhythm stability metrics."""

import numpy as np


def instability_coefficient(cycle_durations):
    """Compute the instability coefficient (IC) of gastric rhythm.

    IC is defined as SD_frequency / mean_frequency, where frequency
    is derived from cycle durations via error propagation:
    ``SD_freq = SD_duration / mean_duration^2``.

    Parameters
    ----------
    cycle_durations : array_like
        Array of cycle durations in seconds (from
        ``gastropy.signal.cycle_durations``).

    Returns
    -------
    float
        Instability coefficient. NaN if fewer than 2 cycles.

    Examples
    --------
    >>> from gastropy.metrics import instability_coefficient
    >>> durs = [20.1, 19.8, 20.3, 19.9, 20.0]
    >>> ic = instability_coefficient(durs)
    """
    durs = np.asarray(cycle_durations, dtype=float)
    durs = durs[~np.isnan(durs)]

    if len(durs) < 2:
        return np.nan

    mean_dur = float(np.mean(durs))
    sd_dur = float(np.std(durs, ddof=1))

    if mean_dur == 0:
        return np.nan

    mean_freq = 1.0 / mean_dur
    sd_freq = sd_dur / (mean_dur**2)

    return sd_freq / mean_freq if mean_freq > 0 else np.nan


def cycle_stats(durations):
    """Compute summary statistics for cycle durations.

    Parameters
    ----------
    durations : array_like
        Array of cycle durations in seconds.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``n_cycles`` : Number of cycles.
        - ``mean_cycle_dur_s`` : Mean duration (seconds).
        - ``sd_cycle_dur_s`` : Standard deviation (seconds, Bessel-corrected).
        - ``prop_within_3sd`` : Proportion of cycles within 3 SD of the mean.
        - ``lower_3sd_s`` : Lower 3-sigma bound (seconds).
        - ``upper_3sd_s`` : Upper 3-sigma bound (seconds).

    Examples
    --------
    >>> from gastropy.metrics import cycle_stats
    >>> stats = cycle_stats([20.1, 19.8, 20.3, 19.9, 20.0])
    >>> stats["mean_cycle_dur_s"]
    20.02
    """
    durs = np.asarray(durations, dtype=float)

    if durs.size == 0:
        return {
            "n_cycles": 0,
            "mean_cycle_dur_s": np.nan,
            "sd_cycle_dur_s": np.nan,
            "prop_within_3sd": np.nan,
            "lower_3sd_s": np.nan,
            "upper_3sd_s": np.nan,
        }

    mean = float(np.mean(durs))
    sd = float(np.std(durs, ddof=1)) if durs.size > 1 else 0.0
    lower = mean - 3.0 * sd
    upper = mean + 3.0 * sd

    prop = 1.0 if sd <= 0 else float(np.mean((durs >= lower) & (durs <= upper)))

    return {
        "n_cycles": int(durs.size),
        "mean_cycle_dur_s": mean,
        "sd_cycle_dur_s": sd,
        "prop_within_3sd": prop,
        "lower_3sd_s": float(lower),
        "upper_3sd_s": float(upper),
    }


def proportion_normogastric(cycle_durations, dur_min=15.0, dur_max=30.0):
    """Compute the proportion of cycles in the normogastric range.

    Normogastric cycles have durations between 15 and 30 seconds
    (corresponding to 2-4 cycles per minute).

    Parameters
    ----------
    cycle_durations : array_like
        Array of cycle durations in seconds.
    dur_min : float, optional
        Minimum normogastric cycle duration (seconds). Default is 15.0.
    dur_max : float, optional
        Maximum normogastric cycle duration (seconds). Default is 30.0.

    Returns
    -------
    float
        Proportion of cycles within the normogastric range (0 to 1).
        NaN if no cycles provided.

    Examples
    --------
    >>> from gastropy.metrics import proportion_normogastric
    >>> proportion_normogastric([20.0, 22.0, 10.0, 35.0])
    0.5
    """
    durs = np.asarray(cycle_durations, dtype=float)
    durs = durs[~np.isnan(durs)]

    if len(durs) == 0:
        return np.nan

    normo = np.sum((durs >= dur_min) & (durs <= dur_max))
    return float(normo / len(durs))
