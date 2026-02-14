"""Surrogate PLV computation via circular time-shifting.

Implements the median rotation method for generating a null
distribution of PLV values, preserving the autocorrelation
structure of both signals.

References
----------
Banellis, L., Rebollo, I., Nikolova, N., & Allen, M. (2025).
Stomach-brain coupling indexes a dimensional signature of mental
health. *Nature Mental Health*.

Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J.
(1999). Measuring phase synchrony in brain signals. *Human Brain
Mapping*, 8(4), 194-208.
"""

import numpy as np

from .plv import phase_locking_value


def surrogate_plv(phase_a, phase_b, buffer_samples=None, n_surrogates=None, stat="median", seed=None):
    """Compute surrogate PLV via circular time-shifting.

    Creates a null distribution of PLV by circularly shifting the
    reference phase time series (``phase_b``) and recomputing PLV
    for each shift. This preserves the autocorrelation of both
    signals while destroying the true phase relationship.

    Parameters
    ----------
    phase_a : array_like, shape (n_timepoints,) or (n_timepoints, n_signals)
        Phase time series (e.g., BOLD voxel phases).
    phase_b : array_like, shape (n_timepoints,)
        Reference phase time series (e.g., EGG phase). This signal
        is circularly shifted.
    buffer_samples : int, optional
        Number of samples to exclude from each edge when generating
        shifts. Prevents near-zero shifts that would approximate
        the empirical PLV. Default is ``n_timepoints // 10``
        (10% of signal length).
    n_surrogates : int, optional
        Number of random shifts to use. If ``None`` (default), uses
        all valid shifts (excluding the buffer at each edge).
    stat : {"median", "mean", "all"}
        Summary statistic across surrogates. ``"median"`` (default)
        returns the median surrogate PLV (as in Banellis et al. 2025).
        ``"mean"`` returns the mean. ``"all"`` returns the full array
        of surrogate PLV values.
    seed : int or np.random.Generator, optional
        Random seed for reproducibility when ``n_surrogates`` is set.

    Returns
    -------
    surr_plv : float, np.ndarray
        Surrogate PLV value(s). Shape depends on ``stat`` and input
        dimensions:

        - ``stat="median"`` or ``"mean"``: same shape as
          ``phase_locking_value(phase_a, phase_b)``
        - ``stat="all"``: shape ``(n_shifts,)`` for 1D ``phase_a``,
          or ``(n_shifts, n_signals)`` for 2D ``phase_a``

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> phase_a = rng.uniform(-np.pi, np.pi, 200)
    >>> phase_b = rng.uniform(-np.pi, np.pi, 200)
    >>> surr = surrogate_plv(phase_a, phase_b, seed=42)
    """
    phase_a = np.asarray(phase_a, dtype=float)
    phase_b = np.asarray(phase_b, dtype=float)
    n = phase_b.shape[0]

    if buffer_samples is None:
        buffer_samples = n // 10

    if buffer_samples < 0:
        raise ValueError(f"buffer_samples must be non-negative, got {buffer_samples}")

    # Generate valid shift indices (excluding buffer at each edge)
    min_shift = buffer_samples
    max_shift = n - buffer_samples
    if min_shift >= max_shift:
        raise ValueError(
            f"buffer_samples={buffer_samples} is too large for signal length {n}. Must be less than {n // 2}."
        )

    all_shifts = np.arange(min_shift, max_shift)

    if n_surrogates is not None and n_surrogates < len(all_shifts):
        rng = np.random.default_rng(seed)
        shifts = rng.choice(all_shifts, size=n_surrogates, replace=False)
        shifts.sort()
    else:
        shifts = all_shifts

    # Compute PLV for each circular shift
    surr_plvs = []
    for shift in shifts:
        shifted_b = np.roll(phase_b, int(shift))
        plv = phase_locking_value(phase_a, shifted_b)
        surr_plvs.append(plv)

    surr_plvs = np.array(surr_plvs)

    if stat == "median":
        return np.median(surr_plvs, axis=0)
    elif stat == "mean":
        return np.mean(surr_plvs, axis=0)
    elif stat == "all":
        return surr_plvs
    else:
        raise ValueError(f"stat must be 'median', 'mean', or 'all', got {stat!r}")


def coupling_zscore(empirical_plv, surrogate_plv):
    """Compute z-scored coupling strength (empirical vs. surrogate).

    When ``surrogate_plv`` is a scalar or same shape as ``empirical_plv``
    (from ``stat="median"``), returns the simple difference. When
    ``surrogate_plv`` is a full distribution (from ``stat="all"``),
    returns a proper z-score.

    Parameters
    ----------
    empirical_plv : float or np.ndarray
        Empirical PLV value(s).
    surrogate_plv : float or np.ndarray
        Surrogate PLV value(s). If 2D with shape ``(n_surrogates, ...)``,
        z-score is computed across the first axis.

    Returns
    -------
    z : float or np.ndarray
        Z-scored coupling strength. Positive values indicate stronger
        coupling than expected by chance.

    Examples
    --------
    >>> coupling_zscore(0.3, 0.15)
    0.15
    """
    empirical_plv = np.asarray(empirical_plv, dtype=float)
    surrogate_plv = np.asarray(surrogate_plv, dtype=float)

    # If surrogate is a distribution (2D or has more dims than empirical)
    if surrogate_plv.ndim > empirical_plv.ndim or (
        surrogate_plv.ndim == 1 and empirical_plv.ndim == 0 and surrogate_plv.shape[0] > 1
    ):
        surr_mean = np.mean(surrogate_plv, axis=0)
        surr_std = np.std(surrogate_plv, axis=0, ddof=1)
        # Avoid division by zero
        surr_std = np.where(surr_std > 0, surr_std, 1.0)
        z = (empirical_plv - surr_mean) / surr_std
    else:
        z = empirical_plv - surrogate_plv

    result = float(z) if z.ndim == 0 else z
    return result


__all__ = ["surrogate_plv", "coupling_zscore"]
