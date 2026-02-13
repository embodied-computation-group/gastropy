"""Signal filtering utilities."""

import numpy as np
from scipy import signal as sp_signal


def design_fir_bandpass(low_hz, high_hz, sfreq, f_order=5, transition_width=0.20, window="hann"):
    """Design a FIR bandpass filter with adaptive tap count.

    Tap count is scaled based on the lower cutoff frequency and
    transition width, matching the behavior of MATLAB-based EGG
    analysis pipelines.

    Parameters
    ----------
    low_hz : float
        Lower passband edge in Hz.
    high_hz : float
        Upper passband edge in Hz.
    sfreq : float
        Sampling frequency in Hz.
    f_order : int, optional
        Filter order scaling factor. Default is 5.
    transition_width : float, optional
        Normalized transition width (0 to 1). Default is 0.20.
    window : str, optional
        Window function for FIR design. Default is ``"hann"``.

    Returns
    -------
    b : np.ndarray
        FIR filter coefficients (numerator).
    a : np.ndarray
        Denominator coefficients (always ``[1.0]`` for FIR).

    Examples
    --------
    >>> b, a = design_fir_bandpass(0.03, 0.07, sfreq=10.0)
    >>> len(b)  # number of taps (odd)
    51
    """
    nyq = sfreq / 2.0
    low = max(1e-6, low_hz)
    high = min(high_hz, nyq - 1e-6)
    high = max(high, low + 1e-6)

    lower_bound = max(low, 1e-6)
    base_taps = int(max(8, np.floor(sfreq / lower_bound)))
    scale = 1.0 / max(transition_width, 1e-3)
    numtaps = int(np.ceil(max(5.0, f_order * base_taps * scale)))
    numtaps = min(numtaps, 500)

    if numtaps % 2 == 0:
        numtaps += 1

    b = sp_signal.firwin(numtaps, [low, high], pass_zero=False, fs=sfreq, window=window)
    a = np.array([1.0])
    return b, a


def apply_bandpass(data, sfreq, low_hz, high_hz, method="fir", **kwargs):
    """Apply a zero-phase bandpass filter to a signal.

    Supports FIR (default) and IIR (Butterworth) implementations.
    FIR uses ``filtfilt`` with Gustafsson's method for robust edge
    handling on long kernels.

    Parameters
    ----------
    data : array_like
        Input signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    low_hz : float
        Lower passband edge in Hz.
    high_hz : float
        Upper passband edge in Hz.
    method : str, optional
        Filter method: ``"fir"`` (default) or ``"iir"``.
    **kwargs
        Additional arguments passed to ``design_fir_bandpass`` when
        ``method="fir"`` (e.g., ``f_order``, ``transition_width``,
        ``window``).

    Returns
    -------
    filtered : np.ndarray
        Filtered signal (same length as input).
    info : dict
        Filter metadata (method, number of taps, window, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import apply_bandpass
    >>> t = np.arange(0, 300, 0.1)
    >>> sig = np.sin(2 * np.pi * 0.05 * t) + np.sin(2 * np.pi * 0.5 * t)
    >>> filtered, info = apply_bandpass(sig, sfreq=10.0, low_hz=0.03, high_hz=0.07)
    """
    data = np.asarray(data, dtype=float)
    info = {}

    if method.lower() == "fir":
        fir_kwargs = {k: v for k, v in kwargs.items() if k in ("f_order", "transition_width", "window")}
        b, a = design_fir_bandpass(low_hz, high_hz, sfreq, **fir_kwargs)
        try:
            filtered = sp_signal.filtfilt(b, a, data, method="gust")
            filtfilt_method = "gust"
        except TypeError:
            filtered = sp_signal.filtfilt(b, a, data)
            filtfilt_method = "pad"
        info.update(
            {
                "filter_method": "fir",
                "fir_numtaps": int(len(b)),
                "fir_window": fir_kwargs.get("window", "hann"),
                "filtfilt_method": filtfilt_method,
            }
        )
    elif method.lower() == "iir":
        order = kwargs.get("order", 4)
        sos = sp_signal.butter(order, [low_hz, high_hz], btype="band", fs=sfreq, output="sos")
        filtered = sp_signal.sosfiltfilt(sos, data)
        info.update(
            {
                "filter_method": "iir_butter",
                "butter_order": order,
            }
        )
    else:
        raise ValueError(f"Unknown filter method: {method!r}. Use 'fir' or 'iir'.")

    return filtered, info
