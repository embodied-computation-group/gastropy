"""Morlet wavelet time-frequency representation for EGG signals."""

import numpy as np
from scipy.signal import fftconvolve


def morlet_tfr(data, sfreq, freqs, n_cycles=7):
    """Compute time-frequency power using Morlet wavelet convolution.

    For each target frequency, constructs a complex Morlet wavelet and
    convolves it with the input signal to produce an instantaneous power
    estimate. This is a pure scipy implementation equivalent to MNE's
    ``tfr_array_morlet`` with ``output='power'``.

    Parameters
    ----------
    data : array_like
        Input signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    freqs : array_like
        Frequencies of interest in Hz.
    n_cycles : float or array_like, optional
        Number of cycles in each Morlet wavelet. Controls the
        time-frequency trade-off: more cycles give better frequency
        resolution but worse time resolution. A scalar value applies
        the same number of cycles to all frequencies; an array allows
        per-frequency control. Default is 7.

    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz (same as input).
    times : np.ndarray
        Time values in seconds, length ``n_samples``.
    power : np.ndarray, shape (n_freqs, n_samples)
        Instantaneous power at each frequency and time point.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.timefreq import morlet_tfr
    >>> t = np.arange(0, 300, 0.1)  # 300s at 10 Hz
    >>> sig = np.sin(2 * np.pi * 0.05 * t)
    >>> freqs, times, power = morlet_tfr(sig, sfreq=10.0,
    ...     freqs=np.arange(0.02, 0.1, 0.005))
    """
    data = np.asarray(data, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    n_samples = len(data)
    times = np.arange(n_samples) / sfreq

    # Broadcast n_cycles to per-frequency array
    n_cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), freqs.shape)

    power = np.empty((len(freqs), n_samples), dtype=float)

    for i, (freq, nc) in enumerate(zip(freqs, n_cycles, strict=True)):
        wavelet = _make_morlet(freq, sfreq, nc)
        analytic = fftconvolve(data, wavelet, mode="same")
        power[i] = np.abs(analytic) ** 2

    return freqs, times, power


def _make_morlet(freq, sfreq, n_cycles):
    """Create a complex Morlet wavelet for a single frequency.

    Parameters
    ----------
    freq : float
        Center frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    n_cycles : float
        Number of oscillation cycles in the Gaussian envelope.

    Returns
    -------
    wavelet : np.ndarray
        Complex Morlet wavelet, unit-energy normalized.
    """
    sigma_t = n_cycles / (2.0 * np.pi * freq)

    # Wavelet duration: 5 standard deviations each side
    t = np.arange(-5 * sigma_t, 5 * sigma_t + 1.0 / sfreq, 1.0 / sfreq)

    oscillation = np.exp(2j * np.pi * freq * t)
    gaussian = np.exp(-(t**2) / (2.0 * sigma_t**2))
    wavelet = oscillation * gaussian

    # Normalize to unit energy
    wavelet /= np.sqrt(0.5 * np.sum(np.abs(wavelet) ** 2))

    return wavelet
