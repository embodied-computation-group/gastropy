"""Visualization functions for EGG data and gastric rhythm analysis.

Core EGG plots are adapted from the analysis pipeline described in
Wolpert et al. (2020). fMRI-specific plots are adapted from the
semi_precision analysis codebase.

References
----------
Wolpert, N., Rebollo, I., & Tallon-Baudry, C. (2020).
Electrogastrography for psychophysiological research: Practical
considerations, analysis pipeline, and normative data in a large
sample. *Psychophysiology*, 57, e13599.
"""

import numpy as np

from ..metrics import NORMOGASTRIA

# Default color palette
_CHANNEL_COLORS = [
    "#E74C3C",
    "#2E86AB",
    "#27AE60",
    "#8E44AD",
    "#E67E22",
    "#1ABC9C",
    "#D35400",
    "#2980B9",
]


def _get_or_create_ax(ax):
    """Return existing axes or create a new figure + axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


def _style_ax(ax, xlabel=None, ylabel=None, title=None):
    """Apply consistent styling to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def plot_psd(freqs, psd, band=None, ax=None, ch_names=None, best_idx=None, peak_freq=None):
    """Plot power spectral density with gastric band shading.

    Supports single-channel (1D ``psd``) or multi-channel (2D ``psd``)
    overlays with automatic channel coloring.

    Parameters
    ----------
    freqs : array_like
        Frequency values in Hz.
    psd : array_like
        PSD values. Shape ``(n_freqs,)`` for single channel or
        ``(n_channels, n_freqs)`` for multi-channel.
    band : GastricBand, optional
        Band to shade. Default is ``NORMOGASTRIA``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    ch_names : list of str, optional
        Channel labels for the legend (multi-channel only).
    best_idx : int, optional
        Index of the best channel to highlight with a thicker line.
    peak_freq : float, optional
        Frequency of the peak to mark with a star.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if band is None:
        band = NORMOGASTRIA

    freqs = np.asarray(freqs)
    psd = np.asarray(psd)

    fig, ax = _get_or_create_ax(ax)

    # Normalize to 2D: (n_channels, n_freqs)
    if psd.ndim == 1:
        psd = psd[np.newaxis, :]

    n_channels = psd.shape[0]

    for i in range(n_channels):
        color = _CHANNEL_COLORS[i % len(_CHANNEL_COLORS)]
        lw = 2.5 if (best_idx is not None and i == best_idx) else 1.2
        label = ch_names[i] if ch_names is not None else None
        ax.plot(freqs, psd[i], color=color, linewidth=lw, label=label)

    # Shade the target band
    ax.axvspan(band.f_lo, band.f_hi, alpha=0.08, color="grey", label=f"{band.name} band")

    # Mark peak frequency
    if peak_freq is not None and best_idx is not None:
        mask = np.argmin(np.abs(freqs - peak_freq))
        ax.plot(
            peak_freq, psd[best_idx, mask], "*", color=_CHANNEL_COLORS[best_idx % len(_CHANNEL_COLORS)], markersize=14
        )
    elif peak_freq is not None:
        mask = np.argmin(np.abs(freqs - peak_freq))
        ax.plot(peak_freq, psd[0, mask], "*", color=_CHANNEL_COLORS[0], markersize=14)

    if ch_names is not None:
        ax.legend(fontsize=8, loc="upper right")

    _style_ax(ax, xlabel="Frequency (Hz)", ylabel="Power", title="Power Spectrum")
    return fig, ax


def plot_egg_overview(signals_df, sfreq, title=None):
    """Plot a 4-panel EGG overview: raw, filtered, phase, amplitude.

    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with columns ``raw``, ``filtered``, ``phase``,
        ``amplitude`` (as returned by ``egg_process``).
    sfreq : float
        Sampling frequency in Hz.
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
        Array of 4 axes.
    """
    import matplotlib.pyplot as plt

    n_samples = len(signals_df)
    times = np.arange(n_samples) / sfreq

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    panels = [
        ("raw", "Raw Signal", "Amplitude"),
        ("filtered", "Filtered", "Amplitude"),
        ("phase", "Phase", "Phase (rad)"),
        ("amplitude", "Amplitude Envelope", "Amplitude"),
    ]

    colors = ["steelblue", "forestgreen", "forestgreen", "coral"]

    for ax, (col, ptitle, ylabel), color in zip(axes, panels, colors, strict=True):
        if col in signals_df.columns:
            data = signals_df[col].values
            # Mean-center raw signal for display
            if col == "raw":
                data = data - np.mean(data)
            ax.plot(times, data, color=color, linewidth=0.8)
        _style_ax(ax, ylabel=ylabel, title=ptitle)

    axes[-1].set_xlabel("Time (seconds)")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


def plot_cycle_histogram(durations, normo_range=(15.0, 30.0), ax=None):
    """Plot histogram of gastric cycle durations.

    Parameters
    ----------
    durations : array_like
        Cycle durations in seconds.
    normo_range : tuple of float, optional
        (min, max) seconds for the normogastric range.
        Default is (15.0, 30.0) corresponding to 2-4 cpm.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    durations = np.asarray(durations, dtype=float)
    durations = durations[~np.isnan(durations)]

    fig, ax = _get_or_create_ax(ax)

    if durations.size > 0:
        ax.hist(durations, bins="auto", color="#27AE60", edgecolor="white", alpha=0.85)

        # Normogastric boundaries
        for bound in normo_range:
            ax.axvline(bound, color="red", linestyle="--", linewidth=2)

        # Compute proportion normogastric
        normo = np.sum((durations >= normo_range[0]) & (durations <= normo_range[1]))
        prop = normo / len(durations) * 100
        _style_ax(
            ax,
            xlabel="Cycle duration (seconds)",
            ylabel="Count",
            title=f"Cycle Duration Distribution ({prop:.1f}% normogastric)",
        )
    else:
        _style_ax(
            ax, xlabel="Cycle duration (seconds)", ylabel="Count", title="Cycle Duration Distribution (no cycles)"
        )

    return fig, ax


def plot_artifacts(phase, times, artifact_info, ax=None):
    """Plot phase time series with artifact regions highlighted.

    Parameters
    ----------
    phase : array_like
        Instantaneous phase in radians.
    times : array_like
        Time values in seconds.
    artifact_info : dict
        Output of ``detect_phase_artifacts``. Must contain
        ``artifact_segments`` key.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    phase = np.asarray(phase)
    times = np.asarray(times)

    fig, ax = _get_or_create_ax(ax)

    ax.plot(times, phase, color="forestgreen", linewidth=0.8)

    # Shade artifact regions in red
    for start_idx, end_idx in artifact_info.get("artifact_segments", []):
        end_idx = min(end_idx, len(times) - 1)
        ax.axvspan(times[start_idx], times[end_idx], alpha=0.3, facecolor="red", edgecolor="none")

    n_art = artifact_info.get("n_artifacts", 0)
    _style_ax(ax, xlabel="Time (seconds)", ylabel="Phase (rad)", title=f"Phase with Artifacts ({n_art} flagged)")

    return fig, ax


def plot_volume_phase(phase_per_vol, tr=None, cut_start=0, cut_end=0, ax=None):
    """Plot per-volume mean phase from fMRI-EGG coupling.

    Parameters
    ----------
    phase_per_vol : array_like
        Mean phase angle per fMRI volume (radians).
    tr : float, optional
        Repetition time in seconds. If provided, x-axis shows time
        instead of volume index.
    cut_start : int, optional
        Number of initial volumes to highlight as cut. Default is 0.
    cut_end : int, optional
        Number of final volumes to highlight as cut. Default is 0.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    phase_per_vol = np.asarray(phase_per_vol, dtype=float)
    n_vols = len(phase_per_vol)

    fig, ax = _get_or_create_ax(ax)

    if tr is not None:
        x = np.arange(n_vols) * tr
        xlabel = "Time (seconds)"
    else:
        x = np.arange(n_vols)
        xlabel = "Volume"

    ax.plot(x, phase_per_vol, ".-", color="steelblue", markersize=3, linewidth=0.8)

    # Highlight cut regions
    if cut_start > 0 and cut_start < n_vols:
        ax.axvspan(x[0], x[min(cut_start, n_vols - 1)], alpha=0.15, color="red", label="Cut")
    if cut_end > 0 and cut_end < n_vols:
        ax.axvspan(x[max(0, n_vols - cut_end)], x[-1], alpha=0.15, color="red")

    _style_ax(ax, xlabel=xlabel, ylabel="Phase (rad)", title="Per-Volume Mean Phase")

    if cut_start > 0 or cut_end > 0:
        ax.legend(fontsize=8)

    return fig, ax


def plot_egg_comprehensive(signals_df, sfreq, phase_per_vol=None, tr=None, artifact_info=None):
    """Create a comprehensive multi-panel EGG analysis figure.

    Combines the overview panels (raw, filtered, phase, amplitude)
    with optional artifact overlay and per-volume phase panel.

    Parameters
    ----------
    signals_df : pd.DataFrame
        DataFrame with columns ``raw``, ``filtered``, ``phase``,
        ``amplitude`` (as returned by ``egg_process``).
    sfreq : float
        Sampling frequency in Hz.
    phase_per_vol : array_like, optional
        Per-volume mean phase (radians) for fMRI data.
    tr : float, optional
        fMRI repetition time in seconds.
    artifact_info : dict, optional
        Output of ``detect_phase_artifacts``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    n_panels = 4
    if phase_per_vol is not None:
        n_panels += 1

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels))
    n_samples = len(signals_df)
    times = np.arange(n_samples) / sfreq

    # Panel 1: Raw (mean-centered)
    raw = signals_df["raw"].values - np.mean(signals_df["raw"].values)
    axes[0].plot(times, raw, color="steelblue", linewidth=0.8)
    _style_ax(axes[0], ylabel="Amplitude", title="Raw Signal")

    # Panel 2: Filtered
    axes[1].plot(times, signals_df["filtered"].values, color="forestgreen", linewidth=0.8)
    _style_ax(axes[1], ylabel="Amplitude", title="Filtered")

    # Panel 3: Phase (with optional artifact overlay)
    axes[2].plot(times, signals_df["phase"].values, color="forestgreen", linewidth=0.8)
    if artifact_info is not None:
        for start_idx, end_idx in artifact_info.get("artifact_segments", []):
            end_idx = min(end_idx, len(times) - 1)
            axes[2].axvspan(times[start_idx], times[end_idx], alpha=0.3, facecolor="red", edgecolor="none")
    n_art = artifact_info.get("n_artifacts", 0) if artifact_info else 0
    art_label = f" ({n_art} artifacts)" if artifact_info else ""
    _style_ax(axes[2], ylabel="Phase (rad)", title=f"Phase{art_label}")

    # Panel 4: Amplitude
    axes[3].plot(times, signals_df["amplitude"].values, color="coral", linewidth=0.8)
    _style_ax(axes[3], ylabel="Amplitude", title="Amplitude Envelope")

    # Panel 5 (optional): Per-volume phase
    if phase_per_vol is not None:
        plot_volume_phase(phase_per_vol, tr=tr, ax=axes[4])

    # Set shared x-label on bottom panel
    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    return fig, axes
