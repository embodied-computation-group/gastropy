"""Phase-locking value (PLV) computation.

Implements the PLV metric for quantifying phase coupling between
two signals (e.g., gastric EGG phase and brain BOLD phase).

References
----------
Lachaux, J. P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999).
Measuring phase synchrony in brain signals. *Human Brain Mapping*, 8(4),
194-208.

Cohen, M. X. (2014). *Analyzing Neural Time Series Data*. MIT Press.
"""

import numpy as np


def phase_locking_value(phase_a, phase_b):
    """Compute the phase-locking value between two phase time series.

    PLV measures the consistency of the phase difference between two
    signals across time. A PLV of 1 indicates perfect phase locking,
    while 0 indicates no consistent phase relationship.

    Parameters
    ----------
    phase_a : array_like, shape (n_timepoints,) or (n_timepoints, n_signals)
        Phase time series in radians. When 2D, PLV is computed between
        each column and ``phase_b``.
    phase_b : array_like, shape (n_timepoints,)
        Reference phase time series in radians. Broadcast against
        columns of ``phase_a`` when ``phase_a`` is 2D.

    Returns
    -------
    plv : float or np.ndarray
        Phase-locking value(s) in [0, 1]. Scalar if both inputs are 1D,
        otherwise array of shape ``(n_signals,)``.

    See Also
    --------
    phase_locking_value_complex : Returns full complex PLV (magnitude + lag).

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(0, 100, 0.1)
    >>> phase_a = 2 * np.pi * 0.05 * t  # constant-frequency phase
    >>> phase_b = phase_a + 0.3          # constant offset = perfect locking
    >>> plv = phase_locking_value(phase_a, phase_b)
    >>> round(plv, 2)
    1.0
    """
    return np.abs(phase_locking_value_complex(phase_a, phase_b))


def phase_locking_value_complex(phase_a, phase_b):
    """Compute the complex phase-locking value.

    Returns the complex mean of the phase difference, from which both
    the PLV magnitude and the preferred phase lag can be extracted.

    Parameters
    ----------
    phase_a : array_like, shape (n_timepoints,) or (n_timepoints, n_signals)
        Phase time series in radians.
    phase_b : array_like, shape (n_timepoints,)
        Reference phase time series in radians.

    Returns
    -------
    cplv : complex or np.ndarray
        Complex PLV. ``abs(cplv)`` gives the PLV magnitude,
        ``np.angle(cplv)`` gives the preferred phase lag.

    Examples
    --------
    >>> import numpy as np
    >>> phase_a = np.zeros(100)
    >>> phase_b = np.full(100, 0.5)
    >>> cplv = phase_locking_value_complex(phase_a, phase_b)
    >>> round(abs(cplv), 2)
    1.0
    >>> round(np.angle(cplv), 2)  # phase lag ~ -0.5
    -0.5
    """
    phase_a = np.asarray(phase_a, dtype=float)
    phase_b = np.asarray(phase_b, dtype=float)

    if phase_a.ndim == 1 and phase_b.ndim == 1:
        if phase_a.shape[0] != phase_b.shape[0]:
            raise ValueError(
                f"phase_a and phase_b must have the same number of timepoints, "
                f"got {phase_a.shape[0]} and {phase_b.shape[0]}"
            )
    elif phase_a.ndim == 2:
        if phase_a.shape[0] != phase_b.shape[0]:
            raise ValueError(
                f"phase_a and phase_b must have the same number of timepoints, "
                f"got {phase_a.shape[0]} and {phase_b.shape[0]}"
            )
        # Broadcast phase_b to match: (n_timepoints,) -> (n_timepoints, 1)
        phase_b = phase_b[:, np.newaxis]
    else:
        raise ValueError(f"phase_a must be 1D or 2D, got {phase_a.ndim}D")

    phase_diff = phase_a - phase_b
    cplv = np.mean(np.exp(1j * phase_diff), axis=0)

    # Return scalar for 1D-1D case
    if cplv.ndim == 0 or (cplv.ndim == 1 and cplv.shape[0] == 1 and phase_a.ndim == 1):
        return complex(cplv)
    return cplv


__all__ = ["phase_locking_value", "phase_locking_value_complex"]
