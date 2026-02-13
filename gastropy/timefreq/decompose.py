"""Per-band time-frequency decomposition of gastric signals."""

import numpy as np

from ..metrics import GASTRIC_BANDS, band_power, cycle_stats, instability_coefficient
from ..signal import apply_bandpass, cycle_durations, instantaneous_phase, psd_welch


def band_decompose(data, sfreq, band, hwhm_hz=0.015, total_range=(0.01, 0.2)):
    """Decompose a signal within a single gastric frequency band.

    Finds the peak frequency in the band from the PSD, applies a
    narrowband FIR filter around that peak, then extracts instantaneous
    phase, amplitude, cycle durations, and summary statistics.

    Parameters
    ----------
    data : array_like
        1D signal (e.g., raw or pre-cleaned EGG).
    sfreq : float
        Sampling frequency in Hz.
    band : GastricBand
        Frequency band to analyse.
    hwhm_hz : float, optional
        Half-width at half-maximum for the narrowband filter (Hz).
        The passband will be ``[peak - hwhm, peak + hwhm]``.
        Default is 0.015.
    total_range : tuple of float, optional
        ``(fmin, fmax)`` for PSD computation and proportional power
        calculation. Default is ``(0.01, 0.2)``.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``band`` : dict with ``name``, ``f_lo``, ``f_hi``.
        - ``peak_freq_hz``, ``max_power``, ``mean_power``,
          ``prop_power``, ``mean_power_ratio`` : PSD-based metrics.
        - ``filtered`` : Narrowband-filtered signal (np.ndarray).
        - ``phase`` : Instantaneous phase in radians (np.ndarray).
        - ``amplitude`` : Instantaneous amplitude envelope (np.ndarray).
        - ``cycle_durations_s`` : Per-cycle durations (np.ndarray).
        - ``cycle_stats`` : dict from :func:`~gastropy.metrics.cycle_stats`.
        - ``instability_coefficient`` : float.
        - ``filter`` : Filter metadata dict.

    See Also
    --------
    multiband_analysis : Run decomposition across multiple bands.
    gastropy.metrics.band_power : PSD band power metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.timefreq import band_decompose
    >>> from gastropy.metrics import NORMOGASTRIA
    >>> sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 0.1))
    >>> result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA)
    >>> result["peak_freq_hz"]
    0.05
    """
    data = np.asarray(data, dtype=float)
    n_samples = len(data)
    times = np.arange(n_samples) / sfreq

    # PSD and band power metrics
    freqs, psd = psd_welch(data, sfreq, fmin=total_range[0], fmax=total_range[1])
    bp = band_power(freqs, psd, band, total_range=total_range)

    result = {
        "band": {"name": band.name, "f_lo": band.f_lo, "f_hi": band.f_hi},
        **bp,
    }

    peak_freq = bp["peak_freq_hz"]

    if not np.isfinite(peak_freq):
        # No valid peak — return NaN placeholders
        result.update(
            {
                "filtered": np.full(n_samples, np.nan),
                "phase": np.full(n_samples, np.nan),
                "amplitude": np.full(n_samples, np.nan),
                "cycle_durations_s": np.array([], dtype=float),
                "cycle_stats": cycle_stats([]),
                "instability_coefficient": np.nan,
                "filter": {},
            }
        )
        return result

    # Narrowband filter around peak
    low_hz = max(1e-6, peak_freq - hwhm_hz)
    high_hz = min(peak_freq + hwhm_hz, sfreq / 2.0 - 1e-6)
    filtered, filter_info = apply_bandpass(data, sfreq, low_hz, high_hz)

    # Phase and amplitude
    phase, analytic = instantaneous_phase(filtered)
    amplitude = np.abs(analytic)

    # Cycle detection and statistics
    durs = cycle_durations(phase, times)
    c_stats = cycle_stats(durs)
    ic = instability_coefficient(durs)

    result.update(
        {
            "filtered": filtered,
            "phase": phase,
            "amplitude": amplitude,
            "cycle_durations_s": durs,
            "cycle_stats": c_stats,
            "instability_coefficient": ic,
            "filter": filter_info,
        }
    )
    return result


def multiband_analysis(data, sfreq, bands=None, hwhm_hz=0.015, total_range=(0.01, 0.2)):
    """Run per-band decomposition across multiple gastric frequency bands.

    For each band, finds the peak frequency, applies a narrowband filter,
    and extracts phase, amplitude, cycle durations, and summary metrics.
    This replicates the per-band time-frequency analysis from the
    semi_precision EGG pipeline.

    Parameters
    ----------
    data : array_like
        1D signal (e.g., raw or pre-cleaned EGG).
    sfreq : float
        Sampling frequency in Hz.
    bands : list of GastricBand, optional
        Bands to analyse. Default is ``GASTRIC_BANDS``
        (bradygastria, normogastria, tachygastria).
    hwhm_hz : float, optional
        Half-width at half-maximum for narrowband filters (Hz).
        Default is 0.015.
    total_range : tuple of float, optional
        ``(fmin, fmax)`` for PSD and proportional power. Default is
        ``(0.01, 0.2)``.

    Returns
    -------
    results : dict
        Dictionary keyed by band name (e.g., ``"brady"``, ``"normo"``,
        ``"tachy"``), each value being the result dict from
        :func:`band_decompose`.

    See Also
    --------
    band_decompose : Single-band decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.timefreq import multiband_analysis
    >>> sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 0.1))
    >>> results = multiband_analysis(sig, sfreq=10.0)
    >>> results["normo"]["peak_freq_hz"]
    0.05
    """
    if bands is None:
        bands = GASTRIC_BANDS

    return {band.name: band_decompose(data, sfreq, band, hwhm_hz=hwhm_hz, total_range=total_range) for band in bands}
