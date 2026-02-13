"""EGG channel and frequency selection."""

import numpy as np

from ..metrics import NORMOGASTRIA
from ..signal import psd_welch


def select_best_channel(data, sfreq, band=None, fmin=0.0, fmax=0.1):
    """Select the EGG channel with the strongest gastric rhythm.

    Ranks channels by peak power in the target frequency band
    (default: normogastria, 2-4 cpm) computed from the unfiltered
    low-frequency PSD.

    Parameters
    ----------
    data : array_like
        Multi-channel EGG data, shape ``(n_channels, n_samples)``.
        For single-channel data, pass shape ``(1, n_samples)``.
    sfreq : float
        Sampling frequency in Hz.
    band : GastricBand, optional
        Frequency band to search for the peak. Default is
        ``NORMOGASTRIA`` (0.033-0.067 Hz / 2-4 cpm).
    fmin : float, optional
        Minimum frequency for PSD computation. Default is 0.0.
    fmax : float, optional
        Maximum frequency for PSD computation. Default is 0.1.

    Returns
    -------
    best_idx : int
        Index of the best channel.
    peak_freq_hz : float
        Peak frequency (Hz) in the target band for the best channel.
    freqs : np.ndarray
        Frequency values from the PSD of the best channel.
    psd : np.ndarray
        PSD values for the best channel.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.egg import select_best_channel
    >>> data = np.random.randn(4, 3000)  # 4 channels, 300s at 10 Hz
    >>> best_idx, peak_freq, freqs, psd = select_best_channel(data, sfreq=10.0)
    """
    if band is None:
        band = NORMOGASTRIA

    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_channels = data.shape[0]

    best_idx = 0
    best_peak = -np.inf
    best_freq = (band.f_lo + band.f_hi) / 2.0
    best_freqs = np.array([], dtype=float)
    best_psd = np.array([], dtype=float)

    for ch_idx in range(n_channels):
        freqs, psd = psd_welch(data[ch_idx], sfreq=sfreq, fmin=fmin, fmax=fmax)
        mask = (freqs >= band.f_lo) & (freqs <= band.f_hi)
        if not np.any(mask):
            continue
        psd_band = psd[mask]
        freqs_band = freqs[mask]
        i_max = int(np.argmax(psd_band))
        peak = psd_band[i_max]
        if peak > best_peak:
            best_peak = float(peak)
            best_idx = int(ch_idx)
            best_freq = float(freqs_band[i_max])
            best_freqs = freqs
            best_psd = psd

    if best_freqs.size == 0:
        best_freqs, best_psd = psd_welch(data[0], sfreq=sfreq, fmin=fmin, fmax=fmax)

    return best_idx, best_freq, best_freqs, best_psd


def select_peak_frequency(data, sfreq, band=None, fmin=0.0, fmax=0.1):
    """Find the peak frequency in the gastric band from a single channel.

    Parameters
    ----------
    data : array_like
        Single-channel EGG signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    band : GastricBand, optional
        Frequency band to search. Default is ``NORMOGASTRIA``.
    fmin : float, optional
        Minimum frequency for PSD computation. Default is 0.0.
    fmax : float, optional
        Maximum frequency for PSD computation. Default is 0.1.

    Returns
    -------
    peak_freq_hz : float
        Peak frequency in the target band (Hz). NaN if no valid peak.
    freqs : np.ndarray
        Frequency values from PSD.
    psd : np.ndarray
        PSD values.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.egg import select_peak_frequency
    >>> sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 0.1))
    >>> peak_freq, freqs, psd = select_peak_frequency(sig, sfreq=10.0)
    """
    if band is None:
        band = NORMOGASTRIA

    data = np.asarray(data, dtype=float)
    freqs, psd = psd_welch(data, sfreq=sfreq, fmin=fmin, fmax=fmax)

    mask = (freqs >= band.f_lo) & (freqs <= band.f_hi)
    if not np.any(mask):
        return np.nan, freqs, psd

    psd_band = psd[mask]
    freqs_band = freqs[mask]
    i_max = int(np.argmax(psd_band))
    return float(freqs_band[i_max]), freqs, psd
