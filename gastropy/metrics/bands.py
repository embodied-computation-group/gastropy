"""Gastric frequency band definitions and band power computation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GastricBand:
    """A named gastric frequency band.

    Parameters
    ----------
    name : str
        Human-readable band name.
    f_lo : float
        Lower frequency bound in Hz.
    f_hi : float
        Upper frequency bound in Hz.
    """

    name: str
    f_lo: float
    f_hi: float

    @property
    def cpm_lo(self):
        """Lower bound in cycles per minute."""
        return self.f_lo * 60.0

    @property
    def cpm_hi(self):
        """Upper bound in cycles per minute."""
        return self.f_hi * 60.0


BRADYGASTRIA = GastricBand("brady", 0.02, 0.03)
"""Bradygastria band: 1-2 cycles per minute (0.02-0.03 Hz)."""

NORMOGASTRIA = GastricBand("normo", 0.03333, 0.06666)
"""Normogastria band: 2-4 cycles per minute (0.033-0.067 Hz)."""

TACHYGASTRIA = GastricBand("tachy", 0.07, 0.17)
"""Tachygastria band: 4-10 cycles per minute (0.07-0.17 Hz)."""

GASTRIC_BANDS = [BRADYGASTRIA, NORMOGASTRIA, TACHYGASTRIA]
"""All three standard gastric frequency bands."""


def band_power(freqs, psd, band, total_range=(0.01, 0.2)):
    """Compute power metrics for a specific frequency band.

    Parameters
    ----------
    freqs : array_like
        Frequency values in Hz (from PSD computation).
    psd : array_like
        Power spectral density values.
    band : GastricBand
        Frequency band to analyze.
    total_range : tuple of float, optional
        ``(f_lo, f_hi)`` defining the total frequency range for
        proportion and ratio calculations. Default is ``(0.01, 0.2)``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``peak_freq_hz`` : Peak frequency in the band (Hz).
        - ``max_power`` : Maximum power in the band.
        - ``mean_power`` : Mean power in the band.
        - ``prop_power`` : Proportion of total power in the band.
        - ``mean_power_ratio`` : Ratio of band mean to total mean power.

        All values are NaN if the band has no frequency coverage.

    Examples
    --------
    >>> from gastropy.signal import psd_welch
    >>> from gastropy.metrics import band_power, NORMOGASTRIA
    >>> freqs, psd = psd_welch(signal, sfreq=10.0, fmin=0.01, fmax=0.2)
    >>> info = band_power(freqs, psd, NORMOGASTRIA)
    >>> info["peak_freq_hz"]
    0.05
    """
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    nan_result = {
        "peak_freq_hz": np.nan,
        "max_power": np.nan,
        "mean_power": np.nan,
        "prop_power": np.nan,
        "mean_power_ratio": np.nan,
    }

    mask_total = (freqs >= total_range[0]) & (freqs <= total_range[1])
    if not np.any(mask_total):
        return nan_result

    mask_band = (freqs >= band.f_lo) & (freqs <= band.f_hi)
    if not np.any(mask_band):
        return nan_result

    psd_band = psd[mask_band]
    freqs_band = freqs[mask_band]
    psd_total = psd[mask_total]

    i_max = int(np.argmax(psd_band))
    peak_freq = float(freqs_band[i_max])
    max_power = float(psd_band[i_max])
    mean_power = float(np.mean(psd_band))

    total_sum = float(np.sum(psd_total))
    prop_power = float(np.sum(psd_band)) / total_sum if total_sum > 0 else np.nan

    total_mean = float(np.mean(psd_total))
    mean_ratio = mean_power / total_mean if total_mean > 0 else np.nan

    return {
        "peak_freq_hz": peak_freq,
        "max_power": max_power,
        "mean_power": mean_power,
        "prop_power": prop_power,
        "mean_power_ratio": mean_ratio,
    }
