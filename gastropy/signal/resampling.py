"""Signal resampling utilities."""

import numpy as np
from scipy import signal as sp_signal


def resample_signal(data, sfreq_orig, sfreq_target):
    """Resample a signal to a new sampling frequency.

    Uses ``scipy.signal.resample`` for polyphase resampling.

    Parameters
    ----------
    data : array_like
        Input signal (1D array).
    sfreq_orig : float
        Original sampling frequency in Hz.
    sfreq_target : float
        Target sampling frequency in Hz.

    Returns
    -------
    resampled : np.ndarray
        Resampled signal.
    sfreq_actual : float
        Actual achieved sampling frequency (may differ slightly from
        target due to integer sample count rounding).

    Raises
    ------
    ValueError
        If the resampled signal would have fewer than 2 samples.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import resample_signal
    >>> sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 0.1))  # 10 Hz
    >>> resampled, actual_rate = resample_signal(sig, 10.0, 2.0)
    """
    data = np.asarray(data, dtype=float)
    n_orig = len(data)
    n_target = int(round(n_orig * sfreq_target / sfreq_orig))

    if n_target < 2:
        raise ValueError(
            f"Resampling from {sfreq_orig} Hz to {sfreq_target} Hz would "
            f"produce only {n_target} sample(s). Need at least 2."
        )

    resampled = sp_signal.resample(data, n_target)
    sfreq_actual = n_target * sfreq_orig / n_orig
    return resampled, sfreq_actual
