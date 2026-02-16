"""
GastroPy Data Module
=====================

Sample datasets for testing, tutorials, and benchmarking.

Sample data is stored in BIDS physio format (``_physio.tsv.gz`` +
``_physio.json`` sidecars).  Use :func:`load_egg` and
:func:`load_fmri_egg` for quick access, or load files directly with
:func:`gastropy.io.read_bids_physio`.

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

_FMRI_SESSIONS = ("0001", "0003", "0004", "0008")

# BIDS filename mapping
_BIDS_FILES = {
    "egg_standalone": "sub-wolpert_task-rest_physio.tsv.gz",
    "fmri_egg_session_0001": "sub-01_ses-0001_task-rest_physio.tsv.gz",
    "fmri_egg_session_0003": "sub-01_ses-0003_task-rest_physio.tsv.gz",
    "fmri_egg_session_0004": "sub-01_ses-0004_task-rest_physio.tsv.gz",
    "fmri_egg_session_0008": "sub-01_ses-0008_task-rest_physio.tsv.gz",
}


def _load_bids(dataset_key):
    """Load a bundled BIDS physio file and return raw dict."""
    from ..io._bids import read_bids_physio

    tsv_path = _DATA_DIR / _BIDS_FILES[dataset_key]
    return read_bids_physio(tsv_path)


def load_fmri_egg(session="0001"):
    """Load a sample fMRI-EGG recording.

    Returns an 8-channel EGG recording collected during fMRI, downsampled
    to 10 Hz, with scanner trigger onset times.

    Parameters
    ----------
    session : str
        Session identifier. Available: ``"0001"``, ``"0003"``, ``"0004"``,
        ``"0008"`` (from the semi_precision study).

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

    raw = _load_bids(f"fmri_egg_session_{session}")

    # Separate EGG channels from trigger column
    columns = raw["columns"]
    trigger_idx = columns.index("trigger")
    egg_indices = [i for i, c in enumerate(columns) if c != "trigger"]

    signal = raw["signal"][egg_indices]
    ch_names = np.array([columns[i] for i in egg_indices])

    # Use exact trigger times from JSON sidecar if available,
    # otherwise reconstruct from the trigger column
    if "TriggerTimesSeconds" in raw:
        trigger_times = np.asarray(raw["TriggerTimesSeconds"], dtype=np.float64)
    else:
        trigger_col = raw["signal"][trigger_idx]
        trigger_samples = np.where(trigger_col > 0.5)[0]
        trigger_times = trigger_samples / raw["sfreq"]

    duration_s = signal.shape[1] / raw["sfreq"]

    return {
        "signal": signal,
        "sfreq": raw["sfreq"],
        "ch_names": ch_names,
        "trigger_times": trigger_times,
        "tr": float(raw.get("TR", 1.856)),
        "duration_s": duration_s,
        "source": str(raw.get("Source", "semi_precision")),
        "session": str(raw.get("Session", session)),
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
    raw = _load_bids("egg_standalone")
    signal = raw["signal"]
    ch_names = np.array(raw["columns"])
    duration_s = signal.shape[1] / raw["sfreq"]

    return {
        "signal": signal,
        "sfreq": raw["sfreq"],
        "ch_names": ch_names,
        "duration_s": duration_s,
        "source": str(raw.get("Source", "wolpert_2020")),
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
     'fmri_egg_session_0004', 'fmri_egg_session_0008',
     'egg_standalone']
    """
    datasets = [f"fmri_egg_session_{s}" for s in _FMRI_SESSIONS]
    datasets.append("egg_standalone")
    return datasets


# ---------------------------------------------------------------------------
# Remote fMRI data (downloaded on demand)
# ---------------------------------------------------------------------------

# Base URL for GitHub Releases
_FMRI_BOLD_BASE_URL = "https://github.com/embodied-computation-group/gastropy/releases/download/sample-data-v1"

# Registry of files with SHA256 hashes: {session: {key: (filename, hash)}}
_FMRI_BOLD_REGISTRY = {
    "0001": {
        "bold": (
            "sub-01_ses-20240815_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "sha256:18949d490dde34af99a00d54d69df041ce7168833cad906f9890a6acf6c41481",
        ),
        "mask": (
            "sub-01_ses-20240815_task-rest_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
            "sha256:adc6c4e30df05db40e37490cabc0a53d844cb72f86970a6bb1db69069d6277b2",
        ),
        "confounds": (
            "sub-01_ses-20240815_task-rest_run-01_desc-confounds_timeseries.tsv",
            "sha256:7736cee95003ceb8f55a763be94f8130ab356eddbcaef18a8451954a95de53d7",
        ),
    },
    "0008": {
        "bold": (
            "sub-01_ses-20241101_task-rest_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
            "sha256:f17edfd40e5770bad7e195168fd0009b1d8fb00e80306042b4c0aad2c894cdc8",
        ),
        "mask": (
            "sub-01_ses-20241101_task-rest_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
            "sha256:73f538f88fe28515012cadb5e19ee0c1feaf5da8f40de77794d028a218f820fd",
        ),
        "confounds": (
            "sub-01_ses-20241101_task-rest_run-01_desc-confounds_timeseries.tsv",
            "sha256:d8f6880b41b996f2a68485226d425629d0d3217c65d1aaeeb0cdfd3023346ae9",
        ),
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
        Session identifier. Available: ``"0001"``, ``"0008"``.
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
        registry={fname: sha for fname, sha in files.values()},
    )

    paths = {}
    for key, (fname, _sha) in files.items():
        paths[key] = Path(fetcher.fetch(fname))

    paths["session"] = session
    paths["tr"] = 1.856

    return paths
