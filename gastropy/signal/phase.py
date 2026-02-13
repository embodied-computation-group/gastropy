"""Phase extraction and cycle detection from signals."""

import numpy as np
from scipy import signal as sp_signal


def instantaneous_phase(data):
    """Compute instantaneous phase via the Hilbert transform.

    Returns the analytic signal (complex) and its phase angle.

    Parameters
    ----------
    data : array_like
        Input signal (1D array, typically bandpass-filtered).

    Returns
    -------
    phase : np.ndarray
        Instantaneous phase in radians (-pi to pi).
    analytic : np.ndarray
        Complex analytic signal from the Hilbert transform.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import instantaneous_phase
    >>> t = np.arange(0, 60, 0.1)
    >>> sig = np.sin(2 * np.pi * 0.05 * t)
    >>> phase, analytic = instantaneous_phase(sig)
    """
    data = np.asarray(data, dtype=float)
    analytic = sp_signal.hilbert(data)
    phase = np.angle(analytic)
    return phase, analytic


def cycle_durations(phase, times):
    """Detect complete cycles from phase and return their durations.

    Locates crossings at each 2pi increment of the unwrapped phase
    using linear interpolation to estimate boundary times. This is
    more robust than simple zero-crossing detection.

    Parameters
    ----------
    phase : array_like
        Instantaneous phase in radians (from ``instantaneous_phase``).
    times : array_like
        Time values corresponding to each phase sample (in seconds).

    Returns
    -------
    durations : np.ndarray
        Duration of each detected cycle in seconds. Empty array if
        fewer than 2 cycle boundaries are found.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import instantaneous_phase, cycle_durations
    >>> sfreq = 10.0
    >>> t = np.arange(0, 300, 1 / sfreq)
    >>> sig = np.sin(2 * np.pi * 0.05 * t)  # 20s cycles
    >>> phase, _ = instantaneous_phase(sig)
    >>> durs = cycle_durations(phase, t)
    """
    phase = np.asarray(phase, dtype=float)
    times = np.asarray(times, dtype=float)

    unwrapped = np.unwrap(phase)
    cycles = (unwrapped - unwrapped[0]) / (2.0 * np.pi)
    k_floor = np.floor(cycles).astype(int)

    idx = np.where(np.diff(k_floor) > 0)[0] + 1
    if idx.size == 0:
        return np.array([], dtype=float)

    boundary_times = []
    for i in idx:
        k_prev = k_floor[i - 1]
        c0, c1 = cycles[i - 1], cycles[i]
        t0, t1 = times[i - 1], times[i]
        target = float(k_prev + 1)
        if c1 == c0:
            t_cross = t1
        else:
            frac = float(np.clip((target - c0) / (c1 - c0), 0.0, 1.0))
            t_cross = float(t0 + frac * (t1 - t0))
        boundary_times.append(t_cross)

    boundary_times = np.array(boundary_times, dtype=float)
    if boundary_times.size < 2:
        return np.array([], dtype=float)

    durations = np.diff(boundary_times)
    return durations[durations > 0]


def mean_phase_per_window(complex_signal, windows):
    """Compute mean phase angle within each time window.

    For each window, computes the mean of the complex analytic signal
    and extracts its phase angle. This is used for per-epoch or
    per-volume phase extraction.

    Parameters
    ----------
    complex_signal : array_like
        Complex analytic signal (from ``instantaneous_phase``).
    windows : list of tuple
        List of ``(start_idx, end_idx)`` pairs defining each window.

    Returns
    -------
    phases : np.ndarray
        Phase angle (radians) per window. NaN for empty or
        out-of-bounds windows.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import instantaneous_phase, mean_phase_per_window
    >>> sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 100, 0.1))
    >>> _, analytic = instantaneous_phase(sig)
    >>> windows = [(0, 50), (50, 100), (100, 150)]
    >>> phases = mean_phase_per_window(analytic, windows)
    """
    complex_signal = np.asarray(complex_signal, dtype=complex)
    n = len(complex_signal)
    phases = []

    for start_idx, end_idx in windows:
        if start_idx >= n or end_idx <= start_idx:
            phases.append(np.nan)
            continue
        segment = complex_signal[start_idx:end_idx]
        if len(segment) == 0:
            phases.append(np.nan)
            continue
        phases.append(np.angle(np.mean(segment)))

    return np.array(phases, dtype=float)
