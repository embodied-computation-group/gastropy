"""Time-domain EGG signal preprocessing functions.

Provides composable artifact removal functions that operate on raw EGG
signals before bandpass filtering. All functions accept 1D (single
channel) or 2D ``(n_channels, n_samples)`` arrays and return the same
shape.

References
----------
Dalmaijer, E. S. (2025). electrography v1.1.1.
https://github.com/esdalmaijer/electrography

Gharibans, A. A., Smarr, B., Kunkel, D. C., Kriegsfeld, L. J.,
Mousa, H., & Coleman, T. P. (2018). Artifact rejection methodology
enables continuous, noninvasive measurement of gastric myoelectric
activity in ambulatory subjects. *Scientific Reports*, 8, 5019.
https://doi.org/10.1038/s41598-018-23302-9
"""

import numpy as np


def hampel_filter(data, k=3, n_sigma=3.0):
    """Remove spike artifacts using the Hampel median identifier.

    Slides a window of length ``2k+1`` across the signal. Any sample
    that deviates from the local median by more than
    ``n_sigma * 1.4826 * MAD`` is replaced by the local median.

    The factor 1.4826 makes the scaled MAD a consistent estimator of
    the standard deviation under a Gaussian assumption
    (Davies & Gather, 1993).

    Parameters
    ----------
    data : array_like
        EGG signal(s). Accepts shape ``(n_samples,)`` or
        ``(n_channels, n_samples)``.
    k : int, optional
        Half-window size: each sample is compared to its ``k``
        neighbours on either side (window length = ``2k+1``).
        Default is 3.
    n_sigma : float, optional
        Outlier threshold in scaled MAD units. Samples further than
        ``n_sigma`` from the local median are replaced. Default is 3.0.

    Returns
    -------
    cleaned : np.ndarray
        Signal with spike outliers replaced by local medians. Same
        shape as input.

    References
    ----------
    Davies, P. L., & Gather, U. (1993). The identification of multiple
    outliers. *Journal of the American Statistical Association*, 88,
    782–792.

    Dalmaijer, E. S. (2025). electrography v1.1.1.
    https://github.com/esdalmaijer/electrography

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import hampel_filter
    >>> sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 0.1))
    >>> sig[50] = 100.0  # inject a spike
    >>> cleaned = hampel_filter(sig)
    >>> np.abs(cleaned[50]) < 5.0  # spike removed
    True
    """
    data = np.asarray(data, dtype=float)
    squeeze = data.ndim == 1
    if squeeze:
        data = data[np.newaxis, :]  # (1, n_samples)

    n_channels, n_samples = data.shape
    signal = data.copy()

    for i in range(n_samples):
        si = max(0, i - k)
        ei = min(n_samples, i + k + 1)
        window = data[:, si:ei]
        med = np.nanmedian(window, axis=1)  # (n_channels,)
        d = np.abs(window - med[:, np.newaxis])
        sd = 1.4826 * np.nanmedian(d, axis=1)  # (n_channels,)
        replace = np.abs(data[:, i] - med) > n_sigma * sd
        signal[replace, i] = med[replace]

    return signal[0] if squeeze else signal


def mad_filter(data, n_sigma=3.0):
    """Remove outliers using a global median absolute deviation filter.

    Computes a single global median and MAD across the entire signal.
    Samples deviating by more than ``n_sigma * 1.4826 * MAD`` are
    replaced by the global median. Faster than ``hampel_filter`` but
    less adaptive to signal drift.

    Parameters
    ----------
    data : array_like
        EGG signal(s). Accepts shape ``(n_samples,)`` or
        ``(n_channels, n_samples)``.
    n_sigma : float, optional
        Outlier threshold in scaled MAD units. Default is 3.0.

    Returns
    -------
    cleaned : np.ndarray
        Signal with global outliers replaced by the median. Same shape
        as input.

    References
    ----------
    Dalmaijer, E. S. (2025). electrography v1.1.1.
    https://github.com/esdalmaijer/electrography

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import mad_filter
    >>> rng = np.random.default_rng(0)
    >>> sig = rng.standard_normal(1000)
    >>> sig[200] = 50.0  # inject a large global outlier
    >>> cleaned = mad_filter(sig)
    >>> np.abs(cleaned[200]) < 5.0
    True
    """
    data = np.asarray(data, dtype=float)
    squeeze = data.ndim == 1
    if squeeze:
        data = data[np.newaxis, :]

    signal = data.copy()
    med = np.nanmedian(signal, axis=1, keepdims=True)  # (n_channels, 1)
    d = np.abs(signal - med)
    sd = 1.4826 * np.nanmedian(d, axis=1)  # (n_channels,)
    threshold = n_sigma * sd

    for ch in range(signal.shape[0]):
        replace = d[ch] > threshold[ch]
        signal[ch, replace] = med[ch, 0]

    return signal[0] if squeeze else signal


def remove_movement_artifacts(data, sfreq, freq=0.05, window=1.0):
    """Attenuate movement artifacts using an LMMSE Wiener filter.

    Estimates local signal variance within sliding windows and computes
    a Wiener-like (LMMSE) correction that suppresses time segments
    dominated by movement noise while preserving the gastric rhythm.

    The algorithm:

    1. For each sample, compute the local mean ``E[y]`` and variance
       ``var_y`` in a window of length ``window / freq`` seconds.
    2. Estimate the signal-of-interest variance ``var_e`` as the
       mean of all local variances.
    3. Compute the predicted noise contribution::

           x_hat = E[y] + clip(var_y - var_e, 0) / max(var_y, var_e)
                   * (y - E[y])

    4. Return the residual ``y - x_hat``.

    Parameters
    ----------
    data : array_like
        EGG signal(s). Accepts shape ``(n_samples,)`` or
        ``(n_channels, n_samples)``.
    sfreq : float
        Sampling frequency in Hz.
    freq : float, optional
        Centre frequency of interest in Hz. Used to set the window
        length (``window / freq`` seconds). Default is 0.05 Hz
        (normogastric centre, 3 cpm).
    window : float, optional
        Window length in cycles of ``freq``. Default is 1.0.

    Returns
    -------
    cleaned : np.ndarray
        Movement-corrected signal. Same shape as input.

    References
    ----------
    Gharibans, A. A., Smarr, B., Kunkel, D. C., Kriegsfeld, L. J.,
    Mousa, H., & Coleman, T. P. (2018). Artifact rejection methodology
    enables continuous, noninvasive measurement of gastric myoelectric
    activity in ambulatory subjects. *Scientific Reports*, 8, 5019.
    https://doi.org/10.1038/s41598-018-23302-9

    Dalmaijer, E. S. (2025). electrography v1.1.1.
    https://github.com/esdalmaijer/electrography

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import remove_movement_artifacts
    >>> t = np.arange(0, 300, 0.1)
    >>> sig = np.sin(2 * np.pi * 0.05 * t)
    >>> cleaned = remove_movement_artifacts(sig, sfreq=10.0)
    >>> cleaned.shape == sig.shape
    True
    """
    data = np.asarray(data, dtype=float)
    squeeze = data.ndim == 1
    if squeeze:
        data = data[np.newaxis, :]

    n_channels, n_samples = data.shape

    # Window length in samples (half-window for centred sliding window)
    win_sec = window / freq
    win_half = int((win_sec * sfreq) // 2)
    si_boundary = win_half
    ei_boundary = n_samples - win_half

    # Pass 1: compute local mean and variance at each sample
    e_y = np.zeros_like(data)
    var_y = np.zeros_like(data)

    for i in range(n_samples):
        si_ = 0 if i < si_boundary else i - win_half
        ei_ = n_samples if i > ei_boundary else i + win_half
        segment = data[:, si_:ei_]
        e_y[:, i] = np.mean(segment, axis=1)
        var_y[:, i] = np.var(segment, axis=1)

    # Estimate signal-of-interest variance as mean of local variances
    var_e = np.mean(var_y, axis=1)  # (n_channels,)

    # Pass 2: compute predicted noise x_hat and subtract
    x_hat = np.zeros_like(data)
    for i in range(n_samples):
        a = var_y[:, i] - var_e
        a = np.maximum(a, 0.0)
        b = np.maximum(var_y[:, i], var_e)
        # When b=0 the signal has zero variance everywhere (e.g. constant
        # signal): no correction needed, so the weight is 0.
        with np.errstate(invalid="ignore", divide="ignore"):
            weight = np.where(b == 0.0, 0.0, a / b)
        x_hat[:, i] = e_y[:, i] + weight * (data[:, i] - e_y[:, i])

    cleaned = data - x_hat
    return cleaned[0] if squeeze else cleaned


__all__ = ["hampel_filter", "mad_filter", "remove_movement_artifacts"]
