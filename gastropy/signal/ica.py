"""ICA-based spatial denoising for multi-channel EGG signals.

Uses FastICA to decompose a multi-channel EGG array into independent
components, retains components whose gastric-band SNR exceeds a
threshold, and reconstructs back to channel space.

References
----------
Dalmaijer, E. S. (2025). electrography v1.1.1.
https://github.com/esdalmaijer/electrography

Hyvärinen, A., & Oja, E. (2000). Independent component analysis:
algorithms and applications. *Neural Networks*, 13, 411–430.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq

from ..metrics import NORMOGASTRIA


def _get_fast_ica():
    """Return FastICA, raising a clear ImportError if sklearn is absent."""
    try:
        from sklearn.decomposition import FastICA

        return FastICA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ica_denoise. Install it with:  pip install 'gastropy[ica]'"
        ) from exc


def _component_snr(component, sfreq, f_lo, f_hi):
    """Compute gastric-band SNR for a single ICA component.

    Uses a Hanning-windowed FFT. Peak power within ``[f_lo, f_hi]``
    is taken as signal; mean power outside is taken as noise.

    Parameters
    ----------
    component : np.ndarray
        1D time series (single ICA component).
    sfreq : float
        Sampling frequency in Hz.
    f_lo, f_hi : float
        Lower and upper bounds of the frequency band of interest.

    Returns
    -------
    snr : float
        Signal-to-noise ratio. Returns 0.0 if noise power is zero.
    """
    n = len(component)
    w = np.hanning(n)
    windowed = component * w
    p = np.abs(rfft(windowed)) * 2.0 / np.sum(w)
    f = rfftfreq(n, 1.0 / sfreq)
    in_band = (f >= f_lo) & (f <= f_hi)
    if not np.any(in_band) or not np.any(~in_band):
        return 0.0
    p_signal = np.nanmax(p[in_band])
    p_noise = np.nanmean(p[~in_band])
    if p_noise == 0.0:
        return 0.0
    return float(p_signal / p_noise)


def ica_denoise(data, sfreq, low_hz=None, high_hz=None, band=None, snr_threshold=3.0, random_state=None):
    """Denoise multi-channel EGG using Independent Component Analysis.

    Decomposes the multi-channel signal into independent components via
    FastICA. Components whose peak-to-mean power ratio within the
    gastric frequency band falls below ``snr_threshold`` are zeroed
    out. The cleaned components are then projected back to the original
    channel space.

    Parameters
    ----------
    data : array_like
        Multi-channel EGG data, shape ``(n_channels, n_samples)``.
        Must have at least 2 channels. For single-channel data use
        standard filtering instead.
    sfreq : float
        Sampling frequency in Hz.
    low_hz : float, optional
        Lower edge of the frequency band of interest (Hz). If None,
        uses ``band.f_lo``.
    high_hz : float, optional
        Upper edge of the frequency band of interest (Hz). If None,
        uses ``band.f_hi``.
    band : GastricBand, optional
        Gastric band supplying default ``low_hz``/``high_hz``.
        Default is ``NORMOGASTRIA`` (2–4 cpm, 0.033–0.067 Hz).
    snr_threshold : float, optional
        Components with gastric-band SNR below this value are removed.
        Default is 3.0.
    random_state : int or None, optional
        Random seed for FastICA reproducibility. Default is None.

    Returns
    -------
    denoised : np.ndarray
        ICA-denoised signal, shape ``(n_channels, n_samples)``.
    info : dict
        Processing metadata:

        - ``n_components`` : int — total number of ICA components.
        - ``n_kept`` : int — components retained above threshold.
        - ``n_removed`` : int — components zeroed out.
        - ``component_snr`` : np.ndarray — SNR for each component.
        - ``snr_threshold`` : float — the threshold used.
        - ``band`` : dict — ``{"f_lo": ..., "f_hi": ...}`` used.

    Raises
    ------
    ImportError
        If scikit-learn is not installed. Install with
        ``pip install 'gastropy[ica]'``.
    ValueError
        If ``data`` is 1-dimensional (requires multi-channel input).
    RuntimeError
        If all ICA components are removed (SNR below threshold for
        every component), indicating the signal may not contain a
        gastric rhythm or the threshold is too strict.

    References
    ----------
    Dalmaijer, E. S. (2025). electrography v1.1.1.
    https://github.com/esdalmaijer/electrography

    Hyvärinen, A., & Oja, E. (2000). Independent component analysis:
    algorithms and applications. *Neural Networks*, 13, 411–430.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import ica_denoise
    >>> rng = np.random.default_rng(42)
    >>> t = np.arange(0, 300, 0.1)
    >>> gastric = np.sin(2 * np.pi * 0.05 * t)
    >>> data = np.stack([gastric + 0.2 * rng.standard_normal(len(t)),
    ...                  gastric + 0.2 * rng.standard_normal(len(t))])
    >>> denoised, info = ica_denoise(data, sfreq=10.0)
    >>> denoised.shape == data.shape
    True
    >>> info["n_kept"] >= 1
    True
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        raise ValueError(
            "ica_denoise requires multi-channel input with shape "
            "(n_channels, n_samples). For single-channel data use "
            "apply_bandpass or egg_clean instead."
        )

    if band is None:
        band = NORMOGASTRIA
    if low_hz is None:
        low_hz = band.f_lo
    if high_hz is None:
        high_hz = band.f_hi

    n_channels = data.shape[0]

    # FastICA expects (n_samples, n_features) — transpose in/out
    FastICA = _get_fast_ica()
    ica = FastICA(random_state=random_state)
    components = ica.fit_transform(data.T).T  # (n_components, n_samples)

    # Score each component and zero out below-threshold ones
    snrs = np.array([_component_snr(components[i], sfreq, low_hz, high_hz) for i in range(n_channels)])
    keep = snrs >= snr_threshold
    n_kept = int(np.sum(keep))
    n_removed = int(np.sum(~keep))

    if n_kept == 0:
        raise RuntimeError(
            f"All {n_channels} ICA components had SNR below {snr_threshold}. "
            "The signal may not contain a detectable gastric rhythm, or the "
            "snr_threshold may be too strict. Consider lowering snr_threshold "
            "or checking your frequency band settings."
        )

    filtered_components = components.copy()
    filtered_components[~keep] = 0.0

    # Reconstruct back to channel space
    denoised = np.ascontiguousarray(ica.inverse_transform(filtered_components.T).T)

    info = {
        "n_components": n_channels,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "component_snr": snrs,
        "snr_threshold": snr_threshold,
        "band": {"f_lo": low_hz, "f_hi": high_hz},
    }

    return denoised, info


__all__ = ["ica_denoise"]
