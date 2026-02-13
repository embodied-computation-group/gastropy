"""EGG signal processing pipeline."""

import numpy as np
import pandas as pd

from ..metrics import NORMOGASTRIA, band_power, cycle_stats, instability_coefficient, proportion_normogastric
from ..signal import apply_bandpass, cycle_durations, instantaneous_phase
from .egg_select import select_peak_frequency


def egg_clean(data, sfreq, low_hz=None, high_hz=None, method="fir", band=None, **filter_kwargs):
    """Clean an EGG signal by applying a bandpass filter.

    By default, filters to the normogastric band (0.033-0.067 Hz,
    2-4 cpm). Custom frequency bounds can be provided.

    Parameters
    ----------
    data : array_like
        Raw EGG signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    low_hz : float, optional
        Lower passband edge in Hz. If None, uses ``band.f_lo``.
    high_hz : float, optional
        Upper passband edge in Hz. If None, uses ``band.f_hi``.
    method : str, optional
        Filter method: ``"fir"`` (default) or ``"iir"``.
    band : GastricBand, optional
        Gastric band for default frequency limits. Default is
        ``NORMOGASTRIA``.
    **filter_kwargs
        Additional arguments passed to ``apply_bandpass`` (e.g.,
        ``f_order``, ``transition_width``).

    Returns
    -------
    cleaned : np.ndarray
        Filtered EGG signal.
    info : dict
        Filter metadata.

    See Also
    --------
    egg_process : Full EGG processing pipeline.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.egg import egg_clean
    >>> sig = np.random.randn(3000)  # 300s at 10 Hz
    >>> cleaned, info = egg_clean(sig, sfreq=10.0)
    """
    if band is None:
        band = NORMOGASTRIA
    if low_hz is None:
        low_hz = band.f_lo
    if high_hz is None:
        high_hz = band.f_hi

    return apply_bandpass(data, sfreq, low_hz, high_hz, method=method, **filter_kwargs)


def egg_process(data, sfreq, band=None, method="fir", **filter_kwargs):
    """Process an EGG signal through the full analysis pipeline.

    Applies bandpass filtering, Hilbert phase extraction, cycle
    detection, and metric computation. Returns a DataFrame of
    processed signals and a metadata dictionary.

    Parameters
    ----------
    data : array_like
        Raw EGG signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    band : GastricBand, optional
        Target gastric band. Default is ``NORMOGASTRIA``.
    method : str, optional
        Filter method: ``"fir"`` (default) or ``"iir"``.
    **filter_kwargs
        Additional arguments passed to ``egg_clean``.

    Returns
    -------
    signals : pd.DataFrame
        DataFrame with columns:

        - ``raw`` : Original signal.
        - ``filtered`` : Bandpass-filtered signal.
        - ``phase`` : Instantaneous phase (radians).
        - ``amplitude`` : Instantaneous amplitude envelope.
    info : dict
        Processing metadata including:

        - ``sfreq`` : Sampling frequency.
        - ``filter`` : Filter parameters.
        - ``peak_freq_hz`` : Peak frequency in the target band.
        - ``cycle_durations_s`` : Detected cycle durations.
        - ``cycle_stats`` : Cycle duration statistics.
        - ``instability_coefficient`` : IC value.
        - ``proportion_normogastric`` : Fraction of normogastric cycles.
        - ``band_power`` : Power metrics for the target band.

    See Also
    --------
    egg_clean : Bandpass filter an EGG signal.

    Examples
    --------
    >>> import numpy as np
    >>> import gastropy as gp
    >>> t = np.arange(0, 300, 0.1)
    >>> sig = np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.random.randn(len(t))
    >>> signals, info = gp.egg_process(sig, sfreq=10.0)
    >>> info["peak_freq_hz"]
    0.05
    """
    if band is None:
        band = NORMOGASTRIA

    data = np.asarray(data, dtype=float)
    n_samples = len(data)
    times = np.arange(n_samples) / sfreq

    # Step 1: Find peak frequency in unfiltered PSD
    peak_freq_hz, psd_freqs, psd_values = select_peak_frequency(data, sfreq, band=band)

    # Step 2: Bandpass filter
    filtered, filter_info = egg_clean(data, sfreq, band=band, method=method, **filter_kwargs)

    # Step 3: Hilbert transform → phase and amplitude
    phase, analytic = instantaneous_phase(filtered)
    amplitude = np.abs(analytic)

    # Step 4: Detect cycles and compute durations
    durs = cycle_durations(phase, times)

    # Step 5: Compute metrics
    c_stats = cycle_stats(durs)
    ic = instability_coefficient(durs)
    prop_normo = proportion_normogastric(durs)
    bp = band_power(psd_freqs, psd_values, band)

    # Build output DataFrame
    signals = pd.DataFrame({
        "raw": data,
        "filtered": filtered,
        "phase": phase,
        "amplitude": amplitude,
    })

    info = {
        "sfreq": sfreq,
        "band": {"name": band.name, "f_lo": band.f_lo, "f_hi": band.f_hi},
        "filter": filter_info,
        "peak_freq_hz": peak_freq_hz,
        "cycle_durations_s": durs,
        "cycle_stats": c_stats,
        "instability_coefficient": ic,
        "proportion_normogastric": prop_normo,
        "band_power": bp,
    }

    return signals, info
