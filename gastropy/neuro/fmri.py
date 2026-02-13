"""fMRI-specific utilities for EGG-fMRI gastric-brain coupling.

These functions handle aspects specific to EGG data recorded
concurrently with fMRI, including scanner trigger parsing,
volume windowing, and transient volume removal.
"""

import numpy as np

from ..signal.phase import mean_phase_per_window


def find_scanner_triggers(annotations, label="R128"):
    """Extract fMRI scanner trigger onset times from MNE annotations.

    Searches for annotations matching the given label (typically
    R128 for volume triggers) and returns their onset times.

    Parameters
    ----------
    annotations : mne.Annotations
        MNE annotations object from a Raw file.
    label : str, optional
        Trigger label to search for. Default is ``"R128"``.
        Matches exact label, ``"*/R128"`` suffixes, or substring.

    Returns
    -------
    onsets : np.ndarray
        Sorted array of trigger onset times in seconds. Empty if
        no matching annotations found.

    Examples
    --------
    >>> import mne
    >>> raw = mne.io.read_raw_fif("egg_data.fif")
    >>> onsets = find_scanner_triggers(raw.annotations, label="R128")
    """
    matched = []
    for desc, onset in zip(annotations.description, annotations.onset, strict=False):
        if desc == label or desc.endswith(f"/{label}") or label in desc:
            matched.append(float(onset))

    return np.sort(np.array(matched, dtype=float)) if matched else np.array([], dtype=float)


def create_volume_windows(onsets, tr, n_volumes):
    """Create volume window index pairs from trigger onsets.

    Maps trigger onset times to ``(start_idx, end_idx)`` pairs
    at the fMRI sampling rate (1/TR Hz).

    Parameters
    ----------
    onsets : array_like
        Trigger onset times in seconds (from ``find_scanner_triggers``).
    tr : float
        Repetition time in seconds.
    n_volumes : int
        Expected number of volumes (windows are capped at this count).

    Returns
    -------
    windows : list of tuple
        List of ``(start_idx, end_idx)`` pairs for each volume.

    Examples
    --------
    >>> onsets = np.arange(0, 100, 1.856)  # ~54 volumes
    >>> windows = create_volume_windows(onsets, tr=1.856, n_volumes=50)
    """
    onsets = np.asarray(onsets, dtype=float)
    windows = []
    n_take = min(int(n_volumes), onsets.size)

    for k in range(n_take):
        t_start = onsets[k]
        t_end = onsets[k + 1] if k + 1 < onsets.size else t_start + tr
        sample_start = int(round(t_start / tr))
        sample_end = int(round(t_end / tr))
        windows.append((sample_start, sample_end))

    return windows


def phase_per_volume(complex_signal, windows):
    """Extract mean phase angle for each fMRI volume.

    Thin wrapper around ``gastropy.signal.mean_phase_per_window``
    with fMRI-specific naming.

    Parameters
    ----------
    complex_signal : array_like
        Complex analytic signal (from Hilbert transform).
    windows : list of tuple
        Volume windows from ``create_volume_windows``.

    Returns
    -------
    phases : np.ndarray
        Phase angle (radians) per volume.

    See Also
    --------
    gastropy.signal.mean_phase_per_window : Generic windowed phase.
    """
    return mean_phase_per_window(complex_signal, windows)


def apply_volume_cuts(data, begin_cut=21, end_cut=21):
    """Remove transient volumes from the start and end of a time series.

    Standard practice to remove filter transients and boundary
    artifacts from EGG-fMRI data.

    Parameters
    ----------
    data : array_like
        1D array of per-volume values (e.g., phase, amplitude).
    begin_cut : int, optional
        Number of volumes to remove from the start. Default is 21.
    end_cut : int, optional
        Number of volumes to remove from the end. Default is 21.

    Returns
    -------
    trimmed : np.ndarray
        Trimmed array. Empty if cuts exceed data length.

    Examples
    --------
    >>> phases = np.random.randn(420)
    >>> trimmed = apply_volume_cuts(phases, begin_cut=21, end_cut=21)
    >>> len(trimmed)
    378
    """
    data = np.asarray(data)
    if begin_cut + end_cut >= len(data):
        return np.array([], dtype=data.dtype)
    if end_cut == 0:
        return data[begin_cut:]
    return data[begin_cut:-end_cut]


__all__ = [
    "find_scanner_triggers",
    "create_volume_windows",
    "phase_per_volume",
    "apply_volume_cuts",
]
