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
fetch_fmri_bold
    Download preprocessed fMRI BOLD data for coupling analysis.
"""

from pathlib import Path

import numpy as np

__all__ = ["load_fmri_egg", "load_egg", "list_datasets", "fetch_fmri_bold"]

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


# ---------------------------------------------------------------------------
# Remote fMRI data (downloaded on demand)
# ---------------------------------------------------------------------------

# Base URL for GitHub Releases (to be updated when data is uploaded)
_FMRI_BOLD_BASE_URL = "https://github.com/embodied-computation-group/gastropy/releases/download/sample-data-v1"

# Registry of files with SHA256 hashes (to be populated after upload)
_FMRI_BOLD_REGISTRY = {
    "0001": {
        "bold": "sub-01_ses-20240815_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "mask": "sub-01_ses-20240815_task-rest_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
        "confounds": "sub-01_ses-20240815_task-rest_run-01_desc-confounds_timeseries.tsv",
    },
}


def fetch_fmri_bold(session="0001", data_dir=None):
    """Download preprocessed fMRI BOLD data for coupling analysis.

    Downloads a preprocessed BOLD NIfTI file, brain mask, and
    fMRIPrep confounds table from a GitHub Release. Files are
    cached locally after the first download.

    Requires the ``pooch`` package (included in the ``neuro``
    optional dependency group).

    Parameters
    ----------
    session : str
        Session identifier. Available: ``"0001"``.
    data_dir : str or Path, optional
        Directory to store downloaded files. Default uses
        ``pooch``'s OS-appropriate cache directory.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``bold`` : Path — path to preprocessed BOLD NIfTI
        - ``mask`` : Path — path to brain mask NIfTI
        - ``confounds`` : Path — path to confounds TSV
        - ``session`` : str — session identifier
        - ``tr`` : float — repetition time (1.856 s)

    Raises
    ------
    ImportError
        If ``pooch`` is not installed.
    ValueError
        If the session identifier is not recognized.

    Notes
    -----
    The BOLD files are large (~1.2-1.4 GB). The first download may
    take several minutes depending on your connection speed.

    The data comes from the semi_precision study, preprocessed with
    fMRIPrep v25.1.4 in MNI152NLin2009cAsym space.

    Examples
    --------
    >>> import gastropy as gp  # doctest: +SKIP
    >>> data = gp.fetch_fmri_bold()  # doctest: +SKIP
    >>> data["bold"]  # doctest: +SKIP
    PosixPath('/home/user/.cache/gastropy/sub-01_ses-..._bold.nii.gz')
    """
    try:
        import pooch
    except ImportError as err:
        raise ImportError(
            "The 'pooch' package is required for downloading fMRI data. Install it with: pip install gastropy[neuro]"
        ) from err

    session = str(session)
    if session not in _FMRI_BOLD_REGISTRY:
        available = ", ".join(repr(s) for s in _FMRI_BOLD_REGISTRY)
        raise ValueError(f"Unknown session {session!r}. Available: {available}")

    files = _FMRI_BOLD_REGISTRY[session]

    if data_dir is not None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    fetcher = pooch.create(
        path=str(data_dir) if data_dir else pooch.os_cache("gastropy"),
        base_url=_FMRI_BOLD_BASE_URL + "/",
        registry={fname: None for fname in files.values()},
    )

    paths = {}
    for key, fname in files.items():
        paths[key] = Path(fetcher.fetch(fname))

    paths["session"] = session
    paths["tr"] = 1.856

    return paths
