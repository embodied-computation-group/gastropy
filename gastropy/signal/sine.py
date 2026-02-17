"""Sine wave fitting for EGG signal characterisation.

Least-squares fitting of a single sine component ``A·sin(2πft + φ)``
to an EGG signal. Useful for quantifying the dominant gastric
frequency, phase, and amplitude from a cleaned signal.

References
----------
Dalmaijer, E. S. (2025). electrography v1.1.1.
https://github.com/esdalmaijer/electrography
"""

import numpy as np
from scipy.optimize import minimize


def sine_model(t, freq, phase, amp):
    """Evaluate a sine wave.

    Computes ``amp * sin(2 * pi * freq * t + phase)``.

    Parameters
    ----------
    t : array_like
        Time values in seconds.
    freq : float
        Frequency in Hz.
    phase : float
        Phase offset in radians.
    amp : float
        Amplitude.

    Returns
    -------
    y : np.ndarray
        Sine wave values at times ``t``.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import sine_model
    >>> t = np.linspace(0, 1, 100)
    >>> y = sine_model(t, freq=0.05, phase=0.0, amp=1.0)
    >>> y.shape
    (100,)
    """
    t = np.asarray(t, dtype=float)
    return amp * np.sin(2.0 * np.pi * freq * t + phase)


def fit_sine(signal, sfreq, freq=None, freq_init=0.05):
    """Fit a sine wave to an EGG signal using least-squares optimisation.

    Minimises the sum of squared residuals between ``signal`` and a
    model ``A·sin(2πft + φ)`` using L-BFGS-B. If ``freq`` is provided,
    only phase and amplitude are fitted (faster); otherwise frequency,
    phase, and amplitude are all free parameters.

    Parameters
    ----------
    signal : array_like
        Single-channel EGG signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    freq : float or None, optional
        If provided, the frequency is fixed to this value (Hz) and only
        phase and amplitude are optimised. If None, all three parameters
        are fitted jointly. Default is None.
    freq_init : float, optional
        Initial frequency guess (Hz) used when ``freq=None``. Defaults
        to 0.05 Hz (normogastric centre, ~3 cpm). Set this to a value
        near the expected dominant frequency when analysing signals
        outside the normogastric band (e.g. ``freq_init=0.1`` for
        tachygastric recordings).

    Returns
    -------
    result : dict
        Fitted parameters:

        - ``freq_hz`` : float — fitted (or fixed) frequency in Hz.
        - ``phase_rad`` : float — fitted phase in radians.
        - ``amplitude`` : float — fitted amplitude (may be negative;
          a negative amplitude is equivalent to a positive amplitude
          with a phase shift of π).
        - ``residual`` : float — sum of squared residuals at solution.

    References
    ----------
    Dalmaijer, E. S. (2025). electrography v1.1.1.
    https://github.com/esdalmaijer/electrography

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import fit_sine
    >>> t = np.arange(0, 300, 0.1)
    >>> sig = 2.5 * np.sin(2 * np.pi * 0.05 * t + 0.3)
    >>> result = fit_sine(sig, sfreq=10.0, freq=0.05)
    >>> abs(result["amplitude"] - 2.5) < 0.1
    True
    """
    signal = np.asarray(signal, dtype=float)
    n_samples = len(signal)
    t = np.arange(n_samples) / sfreq

    def _residuals(betas):
        if len(betas) == 3:
            f, ph, a = betas
        else:
            f = freq
            ph, a = betas
        y_pred = sine_model(t, f, ph, a)
        return float(np.nansum((signal - y_pred) ** 2))

    if freq is None:
        x0 = [freq_init, 0.0, float(np.std(signal))]
        bounds = [(1e-6, None), (-np.pi, np.pi), (None, None)]
    else:
        x0 = [0.0, float(np.std(signal))]
        bounds = [(-np.pi, np.pi), (None, None)]

    opt = minimize(_residuals, x0, method="L-BFGS-B", bounds=bounds)

    if len(opt.x) == 3:
        f_fit, ph_fit, a_fit = opt.x
    else:
        f_fit = freq
        ph_fit, a_fit = opt.x

    return {
        "freq_hz": float(f_fit),
        "phase_rad": float(ph_fit),
        "amplitude": float(a_fit),
        "residual": float(opt.fun),
    }


__all__ = ["sine_model", "fit_sine"]
