"""
GastroPy Data Module
=====================

Sample datasets for testing, tutorials, and benchmarking.

Functions
---------
load_fmri_egg
    Load sample fMRI-EGG recording (8-channel, with scanner triggers).
load_egg
    Load sample standalone EGG recording (7-channel, no fMRI).
list_datasets
    List available sample datasets.
"""

from pathlib import Path

import numpy as np

__all__ = ["load_fmri_egg", "load_egg", "list_datasets"]

_DATA_DIR = Path(__file__).parent

_FMRI_SESSIONS = ("0001", "0003", "0004")


def _load_npz(filename):
    """Load an .npz file from the data directory."""
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Sample data file not found: {path}")
    return dict(np.load(str(path), allow_pickle=False))


def load_fmri_egg(session="0001"):
    """Load a sample fMRI-EGG recording.

    Returns an 8-channel EGG recording collected during fMRI, downsampled
    to 10 Hz, with scanner trigger onset times.

    Parameters
    ----------
    session : str
        Session identifier. Available: ``"0001"``, ``"0003"``, ``"0004"``
        (three baseline sessions from the semi_precision study).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``signal`` : ndarray, shape (8, n_samples) — EGG data in Volts
        - ``sfreq`` : float — sampling rate (10.0 Hz)
        - ``ch_names`` : ndarray of str — channel labels
        - ``trigger_times`` : ndarray — R128 scanner trigger onsets (seconds)
        - ``tr`` : float — fMRI repetition time (1.856 s)
        - ``duration_s`` : float — recording duration in seconds
        - ``source`` : str — data source identifier
        - ``session`` : str — session identifier

    Examples
    --------
    >>> import gastropy as gp
    >>> data = gp.load_fmri_egg()
    >>> data["signal"].shape
    (8, 7795)
    >>> data["sfreq"]
    10.0
    """
    session = str(session)
    if session not in _FMRI_SESSIONS:
        available = ", ".join(repr(s) for s in _FMRI_SESSIONS)
        raise ValueError(f"Unknown session {session!r}. Available: {available}")

    raw = _load_npz(f"fmri_egg_session_{session}.npz")

    return {
        "signal": raw["signal"],
        "sfreq": float(raw["sfreq"]),
        "ch_names": raw["ch_names"],
        "trigger_times": raw["trigger_times"],
        "tr": float(raw["tr"]),
        "duration_s": float(raw["duration_s"]),
        "source": str(raw["source"]),
        "session": str(raw["session"]),
    }


def load_egg():
    """Load a sample standalone EGG recording.

    Returns a 7-channel EGG recording (no fMRI), downsampled to 10 Hz.
    Data from Wolpert et al. (2020), licensed CC BY-NC-SA 3.0.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``signal`` : ndarray, shape (7, n_samples) — EGG data
        - ``sfreq`` : float — sampling rate (10.0 Hz)
        - ``ch_names`` : ndarray of str — channel labels
        - ``duration_s`` : float — recording duration in seconds
        - ``source`` : str — data source identifier

    References
    ----------
    Wolpert, N., Rebollo, I., & Tallon-Baudry, C. (2020).
    Electrogastrography for psychophysiological research: Practical
    considerations, analysis pipeline, and normative data in a large
    sample. *Psychophysiology*, 57, e13599.

    Examples
    --------
    >>> import gastropy as gp
    >>> data = gp.load_egg()
    >>> data["signal"].shape
    (7, 7580)
    >>> data["sfreq"]
    10.0
    """
    raw = _load_npz("egg_standalone.npz")

    return {
        "signal": raw["signal"],
        "sfreq": float(raw["sfreq"]),
        "ch_names": raw["ch_names"],
        "duration_s": float(raw["duration_s"]),
        "source": str(raw["source"]),
    }


def list_datasets():
    """List available sample datasets.

    Returns
    -------
    list of str
        Dataset identifiers that can be loaded.

    Examples
    --------
    >>> import gastropy as gp
    >>> gp.list_datasets()  # doctest: +NORMALIZE_WHITESPACE
    ['fmri_egg_session_0001', 'fmri_egg_session_0003',
     'fmri_egg_session_0004', 'egg_standalone']
    """
    datasets = [f"fmri_egg_session_{s}" for s in _FMRI_SESSIONS]
    datasets.append("egg_standalone")
    return datasets
