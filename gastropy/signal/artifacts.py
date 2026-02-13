"""Phase-based artifact detection for EGG signals.

Ported from Wolpert et al. (2020) ``detect_EGG_artifacts.m``.

Two criteria identify artifact cycles in the filtered EGG phase:

1. **Non-monotonic phase** — within each cycle, the phase should
   increase monotonically from -pi to +pi. Cycles where this fails
   indicate phase distortions or non-physiological signal.
2. **Duration outliers** — cycles whose duration falls outside
   mean +/- N standard deviations are flagged.

References
----------
Wolpert, N., Rebollo, I., & Tallon-Baudry, C. (2020).
Electrogastrography for psychophysiological research: Practical
considerations, analysis pipeline, and normative data in a large
sample. *Psychophysiology*, 57, e13599.
"""

import numpy as np


def find_cycle_edges(phase):
    """Detect cycle boundaries from a wrapped phase time series.

    Finds indices where the phase wraps from +pi back to -pi,
    indicating the start of a new gastric cycle.

    Parameters
    ----------
    phase : array_like
        Instantaneous phase in radians (-pi to pi).

    Returns
    -------
    edges : np.ndarray
        Sample indices of cycle boundaries (phase wrap points).
        Each edge marks the first sample of a new cycle.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import find_cycle_edges
    >>> phase = np.linspace(-np.pi, np.pi, 100)
    >>> phase = np.tile(phase, 5)  # 5 cycles
    >>> edges = find_cycle_edges(phase)
    """
    phase = np.asarray(phase, dtype=float)
    if phase.size < 2:
        return np.array([], dtype=int)

    # Detect large negative jumps (phase wrap from +pi to -pi).
    # Wolpert uses threshold of -1; we use -pi for robustness
    # against small phase fluctuations.
    diffs = np.diff(phase)
    edges = np.where(diffs < -np.pi)[0] + 1
    return edges


def _find_duration_outliers(durations, sd_threshold=3.0):
    """Find cycles with outlier durations (mean +/- N*SD)."""
    durations = np.asarray(durations, dtype=float)
    if durations.size < 2:
        return np.array([], dtype=int)

    mean_dur = np.mean(durations)
    sd_dur = np.std(durations, ddof=1)
    lower = mean_dur - sd_threshold * sd_dur
    upper = mean_dur + sd_threshold * sd_dur

    outliers = np.where((durations < lower) | (durations > upper))[0]
    return outliers


def _find_nonmonotonic_cycles(phase, edges):
    """Find cycles where the phase does not increase monotonically."""
    phase = np.asarray(phase, dtype=float)
    n_samples = len(phase)

    # Build list of (start, end) index pairs for each cycle
    boundaries = np.concatenate(([0], edges, [n_samples]))
    n_cycles = len(boundaries) - 1

    bad = []
    for i in range(n_cycles):
        start = boundaries[i]
        end = boundaries[i + 1]
        segment = phase[start:end]
        if len(segment) < 2:
            continue
        # Phase should be monotonically non-decreasing within a cycle.
        # Check via np.diff — any negative value means non-monotonic.
        if np.any(np.diff(segment) < 0):
            bad.append(i)

    return np.array(bad, dtype=int)


def detect_phase_artifacts(phase, times, sd_threshold=3.0):
    """Detect artifact cycles in an EGG phase time series.

    Uses two phase-based criteria from Wolpert et al. (2020):

    1. **Non-monotonic phase**: cycles where the phase does not
       increase monotonically from -pi to +pi.
    2. **Duration outliers**: cycles whose duration falls outside
       mean +/- ``sd_threshold`` standard deviations.

    Parameters
    ----------
    phase : array_like
        Instantaneous phase in radians (-pi to pi), typically from
        ``instantaneous_phase`` applied to a bandpass-filtered signal.
    times : array_like
        Time values in seconds corresponding to each phase sample.
    sd_threshold : float, optional
        Number of standard deviations for the duration outlier
        criterion. Default is 3.0.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``artifact_mask`` : ndarray of bool, shape (n_samples,)
          True for samples belonging to an artifact cycle.
        - ``artifact_segments`` : list of (start_idx, end_idx) tuples
          Index ranges of each artifact cycle.
        - ``cycle_edges`` : ndarray of int
          Sample indices of cycle boundaries.
        - ``cycle_durations_s`` : ndarray of float
          Duration of each cycle in seconds.
        - ``n_artifacts`` : int
          Total number of artifact cycles detected.
        - ``duration_outlier_cycles`` : ndarray of int
          Indices of cycles flagged as duration outliers.
        - ``nonmonotonic_cycles`` : ndarray of int
          Indices of cycles flagged as non-monotonic.

    References
    ----------
    Wolpert, N., Rebollo, I., & Tallon-Baudry, C. (2020).
    Electrogastrography for psychophysiological research: Practical
    considerations, analysis pipeline, and normative data in a large
    sample. *Psychophysiology*, 57, e13599.

    Examples
    --------
    >>> import numpy as np
    >>> from gastropy.signal import instantaneous_phase, apply_bandpass
    >>> from gastropy.signal import detect_phase_artifacts
    >>> t = np.arange(0, 300, 0.1)
    >>> sig = np.sin(2 * np.pi * 0.05 * t)
    >>> filtered, _ = apply_bandpass(sig, sfreq=10.0, low_hz=0.03, high_hz=0.07)
    >>> phase, _ = instantaneous_phase(filtered)
    >>> artifacts = detect_phase_artifacts(phase, t)
    >>> artifacts["n_artifacts"]
    0
    """
    phase = np.asarray(phase, dtype=float)
    times = np.asarray(times, dtype=float)
    n_samples = len(phase)

    # Find cycle boundaries
    edges = find_cycle_edges(phase)

    # Build cycle boundary pairs: (start, end) for each cycle
    boundaries = np.concatenate(([0], edges, [n_samples]))
    n_cycles = len(boundaries) - 1

    # Compute cycle durations
    edge_times = times[boundaries[:-1]]
    edge_times_end = np.empty(n_cycles)
    for i in range(n_cycles):
        end_idx = boundaries[i + 1]
        edge_times_end[i] = times[min(end_idx, n_samples - 1)]
    cycle_durations_s = edge_times_end - edge_times

    # Criterion 1: duration outliers
    duration_outliers = _find_duration_outliers(cycle_durations_s, sd_threshold)

    # Criterion 2: non-monotonic phase
    nonmonotonic = _find_nonmonotonic_cycles(phase, edges)

    # Merge: union of both criteria
    bad_cycles = np.unique(np.concatenate([duration_outliers, nonmonotonic]))

    # Build per-sample mask and segment list
    artifact_mask = np.zeros(n_samples, dtype=bool)
    artifact_segments = []
    for cycle_idx in bad_cycles:
        start = boundaries[cycle_idx]
        end = boundaries[cycle_idx + 1]
        artifact_mask[start:end] = True
        artifact_segments.append((int(start), int(end)))

    return {
        "artifact_mask": artifact_mask,
        "artifact_segments": artifact_segments,
        "cycle_edges": edges,
        "cycle_durations_s": cycle_durations_s,
        "n_artifacts": len(bad_cycles),
        "duration_outlier_cycles": duration_outliers,
        "nonmonotonic_cycles": nonmonotonic,
    }
