"""Multi-channel EGG processing pipeline.

Provides :func:`egg_process_multichannel` for processing EGG recordings
with multiple electrodes simultaneously. Three named strategies are
supported, following the neurokit2 convention of citable, attributable
method variants:

- ``"per_channel"`` — process each channel independently.
- ``"best_channel"`` — select the channel with the strongest gastric
  rhythm and process that channel only.
- ``"ica"`` — spatially denoise all channels via ICA before processing
  each channel independently.

References
----------
Dalmaijer, E. S. (2025). electrography v1.1.1.
https://github.com/esdalmaijer/electrography
"""

import numpy as np
import pandas as pd

from ..metrics import NORMOGASTRIA
from ..signal.ica import ica_denoise
from .egg_process import egg_process
from .egg_select import select_best_channel


def egg_process_multichannel(data, sfreq, method="per_channel", filter_method="fir", band=None, **kwargs):
    """Process a multi-channel EGG recording.

    Applies one of three named strategies to ``(n_channels, n_samples)``
    EGG data:

    - ``"per_channel"`` — runs :func:`egg_process` independently on
      each channel. Returns metrics for all channels and identifies
      the channel with the strongest normogastric signal.
    - ``"best_channel"`` — uses :func:`select_best_channel` to choose
      the channel with the highest gastric-band peak power, then runs
      :func:`egg_process` on that channel only.
    - ``"ica"`` — denoises all channels via :func:`ica_denoise`
      (FastICA spatial decomposition) then runs ``"per_channel"`` on
      the reconstructed data.

    Parameters
    ----------
    data : array_like
        Multi-channel EGG data, shape ``(n_channels, n_samples)``.
        Must have at least 2 channels; pass a 2D array even for a
        single channel (shape ``(1, n_samples)``).
    sfreq : float
        Sampling frequency in Hz.
    method : str, optional
        Processing strategy: ``"per_channel"`` (default),
        ``"best_channel"``, or ``"ica"``.
    filter_method : str, optional
        Filter method passed to :func:`egg_process` for each channel:
        ``"fir"`` (default), ``"iir"``, or ``"dalmaijer2025"``.
        Kept separate from ``method`` (the multichannel strategy) to
        avoid a parameter name collision.
    band : GastricBand, optional
        Target gastric band. Default is ``NORMOGASTRIA``.
    **kwargs
        Additional keyword arguments passed to :func:`egg_process`.
        For ``"ica"``, also accepts ``ica_low_hz``, ``ica_high_hz``,
        ``ica_snr_threshold``, and ``ica_random_state`` to control
        the ICA denoising step.

    Returns
    -------
    result : dict
        For ``"per_channel"`` and ``"ica"``:

        - ``"channels"`` : dict mapping channel index (int) →
          ``(signals_df, info)`` tuples, one per channel.
        - ``"best_idx"`` : int — index of the channel with the
          highest normogastric band power.
        - ``"summary"`` : pd.DataFrame — one row per channel with
          columns ``channel``, ``peak_freq_hz``,
          ``instability_coefficient``, ``proportion_normogastric``,
          and ``band_power_mean``.
        - ``"method"`` : str — the method used.
        - ``"ica_info"`` : dict — ICA metadata (only for ``"ica"``).

        For ``"best_channel"``:

        - ``"signals"`` : pd.DataFrame — as returned by
          :func:`egg_process`.
        - ``"info"`` : dict — as returned by :func:`egg_process`,
          with ``"best_channel_idx"`` added.
        - ``"method"`` : str — ``"best_channel"``.

    Raises
    ------
    ValueError
        If ``data`` is 1-dimensional or has fewer than 2 channels.
    ValueError
        If ``method`` is not one of the supported options.

    Notes
    -----
    ``best_idx`` in the ``"per_channel"`` and ``"ica"`` results is
    determined by ``band_power_mean`` from the post-processing output
    of :func:`egg_process`.  The ``"best_channel"`` strategy uses
    :func:`select_best_channel`, which ranks by PSD **peak** power
    before processing.  These two criteria may occasionally disagree
    on noisy data.

    See Also
    --------
    egg_process : Single-channel EGG processing pipeline.
    select_best_channel : Channel ranking by gastric-band peak power.
    ica_denoise : ICA-based spatial denoising.

    References
    ----------
    Dalmaijer, E. S. (2025). electrography v1.1.1.
    https://github.com/esdalmaijer/electrography

    Examples
    --------
    >>> import numpy as np
    >>> import gastropy as gp
    >>> rng = np.random.default_rng(0)
    >>> t = np.arange(0, 300, 0.1)
    >>> gastric = np.sin(2 * np.pi * 0.05 * t)
    >>> data = np.stack([gastric + 0.1 * rng.standard_normal(len(t)),
    ...                  gastric + 0.1 * rng.standard_normal(len(t))])
    >>> result = gp.egg_process_multichannel(data, sfreq=10.0)
    >>> sorted(result.keys())
    ['best_idx', 'channels', 'method', 'summary']
    """
    data = np.asarray(data, dtype=float)

    if data.ndim == 1:
        raise ValueError(
            "egg_process_multichannel requires 2D input with shape "
            "(n_channels, n_samples). For single-channel processing use "
            "egg_process instead."
        )
    if data.shape[0] < 2:
        raise ValueError(
            "egg_process_multichannel requires at least 2 channels. "
            f"Got shape {data.shape}. For single-channel data use egg_process."
        )

    if band is None:
        band = NORMOGASTRIA

    valid_methods = ("per_channel", "best_channel", "ica")
    if method not in valid_methods:
        raise ValueError(f"Unknown method {method!r}. Choose from {valid_methods}.")

    if method == "best_channel":
        return _process_best_channel(data, sfreq, band, filter_method=filter_method, **kwargs)
    elif method == "ica":
        return _process_ica(data, sfreq, band, filter_method=filter_method, **kwargs)
    else:
        return _process_per_channel(data, sfreq, band, filter_method=filter_method, **kwargs)


def _process_per_channel(data, sfreq, band, filter_method="fir", **kwargs):
    """Run egg_process on each channel independently."""
    n_channels = data.shape[0]
    channels = {}
    for ch_idx in range(n_channels):
        signals_df, info = egg_process(data[ch_idx], sfreq, band=band, method=filter_method, **kwargs)
        channels[ch_idx] = (signals_df, info)

    summary, best_idx = _build_summary(channels, n_channels, band)
    return {
        "channels": channels,
        "best_idx": best_idx,
        "summary": summary,
        "method": "per_channel",
    }


def _process_best_channel(data, sfreq, band, filter_method="fir", **kwargs):
    """Select strongest channel and run egg_process on it."""
    best_idx, peak_freq, freqs, psd = select_best_channel(data, sfreq, band=band)
    signals_df, info = egg_process(data[best_idx], sfreq, band=band, method=filter_method, **kwargs)
    info["best_channel_idx"] = int(best_idx)
    return {
        "signals": signals_df,
        "info": info,
        "method": "best_channel",
    }


def _process_ica(data, sfreq, band, filter_method="fir", **kwargs):
    """ICA-denoise then run per-channel processing."""
    # Extract ICA-specific kwargs (not passed to egg_process)
    ica_low_hz = kwargs.pop("ica_low_hz", band.f_lo)
    ica_high_hz = kwargs.pop("ica_high_hz", band.f_hi)
    ica_snr_threshold = kwargs.pop("ica_snr_threshold", 3.0)
    random_state = kwargs.pop("ica_random_state", None)

    denoised, ica_info = ica_denoise(
        data,
        sfreq,
        low_hz=ica_low_hz,
        high_hz=ica_high_hz,
        snr_threshold=ica_snr_threshold,
        random_state=random_state,
    )

    result = _process_per_channel(denoised, sfreq, band, filter_method=filter_method, **kwargs)
    result["method"] = "ica"
    result["ica_info"] = ica_info
    return result


def _build_summary(channels, n_channels, band):
    """Build a per-channel summary DataFrame and find the best channel."""
    rows = []
    best_idx = 0
    best_power = -np.inf

    for ch_idx in range(n_channels):
        _, info = channels[ch_idx]
        bp = info.get("band_power", {})
        mean_power = bp.get("mean_power", np.nan)
        rows.append(
            {
                "channel": ch_idx,
                "peak_freq_hz": info.get("peak_freq_hz", np.nan),
                "instability_coefficient": info.get("instability_coefficient", np.nan),
                "proportion_normogastric": info.get("proportion_normogastric", np.nan),
                "band_power_mean": mean_power,
            }
        )
        if not np.isnan(mean_power) and mean_power > best_power:
            best_power = mean_power
            best_idx = ch_idx

    summary = pd.DataFrame(rows)
    return summary, best_idx


__all__ = ["egg_process_multichannel"]
