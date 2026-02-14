"""Circular statistics for phase coupling analysis.

Provides circular mean, resultant length, and the Rayleigh test
for non-uniformity of circular data.

References
----------
Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics*.
Wiley.

Fisher, N. I. (1993). *Statistical Analysis of Circular Data*.
Cambridge University Press.
"""

import numpy as np


def circular_mean(phases):
    """Compute the circular (directional) mean of phase values.

    Parameters
    ----------
    phases : array_like
        Phase values in radians.

    Returns
    -------
    mean_phase : float
        Circular mean direction in radians (-pi to pi).

    Examples
    --------
    >>> import numpy as np
    >>> phases = np.array([0.1, 0.2, 0.15, 0.12])
    >>> round(circular_mean(phases), 2)
    0.14
    """
    phases = np.asarray(phases, dtype=float)
    return float(np.angle(np.mean(np.exp(1j * phases))))


def resultant_length(phases):
    """Compute the mean resultant length of phase values.

    The resultant length R measures the concentration of phases
    around their mean direction. R = 1 indicates all phases are
    identical, R = 0 indicates uniform distribution.

    Parameters
    ----------
    phases : array_like
        Phase values in radians.

    Returns
    -------
    R : float
        Mean resultant length in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> phases = np.zeros(100)  # all identical
    >>> resultant_length(phases)
    1.0
    """
    phases = np.asarray(phases, dtype=float)
    return float(np.abs(np.mean(np.exp(1j * phases))))


def rayleigh_test(phases):
    """Rayleigh test for non-uniformity of circular data.

    Tests the null hypothesis that the population is uniformly
    distributed around the circle against the alternative that
    there is a preferred direction (unimodal clustering).

    Parameters
    ----------
    phases : array_like
        Phase values in radians.

    Returns
    -------
    z_stat : float
        Rayleigh's z statistic (z = n * R^2).
    p_value : float
        Approximate p-value.

    References
    ----------
    Wilkie, D. (1983). Rayleigh test for randomness of circular data.
    *Applied Statistics*, 32, 311-312.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> uniform_phases = rng.uniform(-np.pi, np.pi, 200)
    >>> z, p = rayleigh_test(uniform_phases)
    >>> p > 0.05  # fail to reject uniformity
    True
    """
    phases = np.asarray(phases, dtype=float)
    n = len(phases)
    if n == 0:
        return 0.0, 1.0

    R = resultant_length(phases)
    z = n * R**2

    # Approximation from Wilkie (1983), accurate for n >= 10
    p = np.exp(-z) * (
        1.0 + (2.0 * z - z**2) / (4.0 * n) - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2)
    )
    p = max(0.0, min(1.0, float(p)))

    return float(z), p


__all__ = ["circular_mean", "resultant_length", "rayleigh_test"]
