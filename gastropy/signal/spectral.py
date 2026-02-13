"""Power spectral density estimation."""

import numpy as np
from scipy import signal as sp_signal


def psd_welch(data, sfreq, fmin=0.0, fmax=0.1, overlap=0.25):
    """Compute power spectral density using Welch's method.

    Uses 200-second Hann windows and 1000-second zero-padding for
    fine frequency resolution, matching standard EGG analysis
    parameters.

    Parameters
    ----------
    data : array_like
        Input signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    fmin : float, optional
        Minimum frequency to return (Hz). Default is 0.0.
    fmax : float, optional
        Maximum frequency to return (Hz). Default is 0.1.
    overlap : float, optional
        Fraction of segment overlap (0 to 1). Default is 0.25.
        Use 0.75 for smoother estimates on short recordings
        (Wolpert et al. 2020 convention).

    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz, masked to [fmin, fmax].
    psd : np.ndarray
        Power spectral density values, masked to [fmin, fmax].

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import psd_welch
    >>> t = np.arange(0, 300, 0.1)  # 300s at 10 Hz
    >>> sig = np.sin(2 * np.pi * 0.05 * t)  # 0.05 Hz = 3 cpm
    >>> freqs, psd = psd_welch(sig, sfreq=10.0, fmin=0.01, fmax=0.1)
    """
    data = np.asarray(data, dtype=float)

    nperseg = int(round(sfreq * 200.0))
    nperseg = max(nperseg, int(sfreq * 60.0))
    noverlap = int(overlap * nperseg)
    nfft = int(round(sfreq * 1000.0))

    freqs, psd = sp_signal.welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        scaling="spectrum",
        average="mean",
    )
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[mask]
