"""fMRI-specific utilities for EGG-fMRI gastric-brain coupling.

These functions handle aspects specific to EGG data recorded
concurrently with fMRI, including scanner trigger parsing,
volume windowing, transient volume removal, confound regression,
voxelwise phase extraction, and PLV map computation.
"""

import numpy as np
from scipy import signal as sp_signal

from ..coupling.plv import phase_locking_value
from ..coupling.surrogate import surrogate_plv as _surrogate_plv
from ..signal.filtering import apply_bandpass
from ..signal.phase import instantaneous_phase, mean_phase_per_window


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


# ---------------------------------------------------------------------------
# Layer 2: fMRI-EGG coupling convenience functions
# ---------------------------------------------------------------------------

# Default fMRIPrep confound columns: 6 motion + 6 aCompCor
_DEFAULT_CONFOUND_COLS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "a_comp_cor_00",
    "a_comp_cor_01",
    "a_comp_cor_02",
    "a_comp_cor_03",
    "a_comp_cor_04",
    "a_comp_cor_05",
]


def regress_confounds(bold_2d, confounds_df, confound_cols=None):
    """Remove confound signals from BOLD data via GLM regression.

    Performs ordinary least squares regression of confound regressors
    on each voxel's BOLD time series and returns the z-scored residuals.
    This follows the approach used in standard fMRI denoising pipelines.

    Parameters
    ----------
    bold_2d : array_like, shape (n_voxels, n_timepoints)
        BOLD time series for each voxel (e.g., from a masked NIfTI).
    confounds_df : pandas.DataFrame
        Confounds table (e.g., from fMRIPrep ``*_confounds_timeseries.tsv``).
        Must have at least ``n_timepoints`` rows.
    confound_cols : list of str, optional
        Column names to use as regressors. Default uses 6 motion
        parameters + 6 aCompCor components (12 regressors total).

    Returns
    -------
    residuals : np.ndarray, shape (n_voxels, n_timepoints)
        Z-scored residuals after confound regression.

    Notes
    -----
    NaN values in the first row of derivative columns (common in
    fMRIPrep confounds) are replaced with 0 before regression.
    """
    bold_2d = np.asarray(bold_2d, dtype=float)
    if bold_2d.ndim != 2:
        raise ValueError(f"bold_2d must be 2D (n_voxels, n_timepoints), got {bold_2d.ndim}D")

    if confound_cols is None:
        confound_cols = _DEFAULT_CONFOUND_COLS

    # Extract confound matrix and handle NaNs
    X = confounds_df[confound_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    n_voxels, n_time = bold_2d.shape
    if X.shape[0] != n_time:
        raise ValueError(f"Confounds have {X.shape[0]} timepoints but BOLD has {n_time}. They must match.")

    # Add intercept
    X_design = np.column_stack([np.ones(n_time), X])

    # Z-score BOLD before regression
    bold_mean = bold_2d.mean(axis=1, keepdims=True)
    bold_std = bold_2d.std(axis=1, keepdims=True)
    bold_std[bold_std == 0] = 1.0
    bold_z = (bold_2d - bold_mean) / bold_std

    # OLS via least squares (QR-based)
    # Solve: X_design @ betas = bold_z.T  ->  betas shape (n_regressors, n_voxels)
    betas, _, _, _ = np.linalg.lstsq(X_design, bold_z.T, rcond=None)
    predicted = X_design @ betas  # (n_time, n_voxels)
    residuals = bold_z.T - predicted  # (n_time, n_voxels)

    # Z-score residuals
    res_mean = residuals.mean(axis=0, keepdims=True)
    res_std = residuals.std(axis=0, keepdims=True)
    res_std[res_std == 0] = 1.0
    residuals = (residuals - res_mean) / res_std

    return residuals.T  # back to (n_voxels, n_timepoints)


def bold_voxelwise_phases(
    bold_2d, peak_freq_hz, sfreq, hwhm_hz=0.015, begin_cut=0, end_cut=0, filter_method="iir", filter_order=4
):
    """Extract instantaneous phase at the gastric frequency for each voxel.

    Bandpass-filters each voxel's BOLD time series at the individual
    gastric peak frequency, then applies the Hilbert transform to
    extract the instantaneous phase. Optionally trims edge volumes
    to remove filter ringing artifacts.

    The default IIR (Butterworth) filter is used because FIR filters
    require many more taps than available BOLD volumes at typical
    fMRI sampling rates (~0.5 Hz). IIR filtering is vectorized across
    all voxels for performance.

    Parameters
    ----------
    bold_2d : array_like, shape (n_voxels, n_timepoints)
        BOLD time series (typically confound-regressed residuals).
    peak_freq_hz : float
        Individual gastric peak frequency in Hz (e.g., 0.05 for 3 cpm).
    sfreq : float
        BOLD sampling frequency in Hz (1 / TR).
    hwhm_hz : float, optional
        Half-width at half-maximum of the bandpass filter in Hz.
        Default is 0.015 Hz, matching the StomachBrain pipeline.
    begin_cut : int, optional
        Number of timepoints to remove from the start. Default is 0.
    end_cut : int, optional
        Number of timepoints to remove from the end. Default is 0.
    filter_method : str, optional
        ``"iir"`` (default) for Butterworth, ``"fir"`` for FIR.
        IIR is recommended for BOLD data where the number of volumes
        is typically too short for high-order FIR filters.
    filter_order : int, optional
        Butterworth filter order when ``filter_method="iir"``.
        Default is 4.

    Returns
    -------
    phases : np.ndarray, shape (n_voxels, n_timepoints_trimmed)
        Instantaneous phase in radians (-pi to pi) for each voxel.
    """
    bold_2d = np.asarray(bold_2d, dtype=float)
    if bold_2d.ndim != 2:
        raise ValueError(f"bold_2d must be 2D (n_voxels, n_timepoints), got {bold_2d.ndim}D")

    n_voxels, n_time = bold_2d.shape

    # Early exit if cuts would consume all timepoints
    if begin_cut + end_cut >= n_time:
        return np.empty((n_voxels, 0), dtype=float)

    low_hz = max(peak_freq_hz - hwhm_hz, 1e-6)
    high_hz = peak_freq_hz + hwhm_hz

    if filter_method == "iir":
        # Vectorized IIR path: filter + Hilbert on all voxels at once
        sos = sp_signal.butter(filter_order, [low_hz, high_hz], btype="band", fs=sfreq, output="sos")
        filtered = sp_signal.sosfiltfilt(sos, bold_2d, axis=1)
        analytic = sp_signal.hilbert(filtered, axis=1)
        phases = np.angle(analytic)
    else:
        # Per-voxel FIR path (original)
        phases = np.empty_like(bold_2d)
        for i in range(n_voxels):
            filt_i, _ = apply_bandpass(bold_2d[i], sfreq, low_hz, high_hz, method="fir")
            phase_i, _ = instantaneous_phase(filt_i)
            phases[i] = phase_i

    # Apply edge trimming
    if begin_cut > 0 or end_cut > 0:
        if begin_cut + end_cut >= n_time:
            return np.empty((n_voxels, 0), dtype=float)
        phases = phases[:, begin_cut:] if end_cut == 0 else phases[:, begin_cut:-end_cut]

    return phases


def compute_plv_map(egg_phase, bold_phases, vol_shape=None, mask_indices=None, artifact_mask=None):
    """Compute a voxelwise PLV map between EGG and BOLD phases.

    Parameters
    ----------
    egg_phase : array_like, shape (n_timepoints,)
        Gastric phase per volume (from ``phase_per_volume`` +
        ``apply_volume_cuts``).
    bold_phases : array_like, shape (n_voxels, n_timepoints)
        BOLD phase per volume for each voxel (from
        ``bold_voxelwise_phases``).
    vol_shape : tuple of int, optional
        3D volume dimensions (e.g., ``(97, 115, 97)``). If provided
        along with ``mask_indices``, returns a 3D volume.
    mask_indices : array_like, optional
        Boolean mask or integer indices mapping voxels back to the
        3D volume. Required if ``vol_shape`` is provided.
    artifact_mask : array_like of bool, shape (n_timepoints,), optional
        Boolean mask where ``True`` = clean volume, ``False`` = artifact.
        Artifact volumes are excluded from the PLV computation.

    Returns
    -------
    plv : np.ndarray
        PLV values. Shape is ``(n_voxels,)`` if ``vol_shape`` is
        ``None``, otherwise ``vol_shape``.
    """
    egg_phase = np.asarray(egg_phase, dtype=float)
    bold_phases = np.asarray(bold_phases, dtype=float)

    if egg_phase.shape[0] != bold_phases.shape[1]:
        raise ValueError(
            f"egg_phase has {egg_phase.shape[0]} timepoints but "
            f"bold_phases has {bold_phases.shape[1]}. They must match."
        )

    # PLV: (n_timepoints, n_voxels) vs (n_timepoints,)
    plv = phase_locking_value(bold_phases.T, egg_phase, mask=artifact_mask)

    if vol_shape is not None and mask_indices is not None:
        vol = np.zeros(vol_shape, dtype=float)
        vol[mask_indices] = plv
        return vol

    return plv


def compute_surrogate_plv_map(egg_phase, bold_phases, vol_shape=None, mask_indices=None, artifact_mask=None, **kwargs):
    """Compute a surrogate PLV map via circular time-shifting.

    Same as ``compute_plv_map`` but uses surrogate PLV to generate
    a null distribution map for statistical comparison.

    Parameters
    ----------
    egg_phase : array_like, shape (n_timepoints,)
        Gastric phase per volume.
    bold_phases : array_like, shape (n_voxels, n_timepoints)
        BOLD phase per volume for each voxel.
    vol_shape : tuple of int, optional
        3D volume dimensions for reshaping output.
    mask_indices : array_like, optional
        Mask indices for 3D volume reconstruction.
    artifact_mask : array_like of bool, shape (n_timepoints,), optional
        Boolean mask where ``True`` = clean volume, ``False`` = artifact.
        Artifact volumes are excluded from the surrogate PLV computation.
    **kwargs
        Additional arguments passed to ``surrogate_plv``
        (e.g., ``buffer_samples``, ``n_surrogates``, ``stat``, ``seed``).

    Returns
    -------
    surr_plv : np.ndarray
        Surrogate PLV values. Shape depends on ``vol_shape`` and
        ``stat`` parameter.

    See Also
    --------
    gastropy.coupling.surrogate_plv : Core surrogate PLV computation.
    """
    egg_phase = np.asarray(egg_phase, dtype=float)
    bold_phases = np.asarray(bold_phases, dtype=float)

    if egg_phase.shape[0] != bold_phases.shape[1]:
        raise ValueError(
            f"egg_phase has {egg_phase.shape[0]} timepoints but "
            f"bold_phases has {bold_phases.shape[1]}. They must match."
        )

    surr = _surrogate_plv(bold_phases.T, egg_phase, mask=artifact_mask, **kwargs)

    if vol_shape is not None and mask_indices is not None and surr.ndim == 1:
        vol = np.zeros(vol_shape, dtype=float)
        vol[mask_indices] = surr
        return vol

    return surr


def artifact_mask_to_volumes(sample_mask, trigger_times, sfreq, tr, begin_cut=21, end_cut=21):
    """Map a sample-level artifact mask to volume-level.

    Converts the per-sample boolean artifact mask (from
    ``detect_phase_artifacts``) to a per-volume boolean mask
    suitable for censoring bad volumes in PLV computation.

    A volume is marked as artifact (``False``) if **any** EGG sample
    within its time window is flagged in the sample mask.

    Parameters
    ----------
    sample_mask : array_like of bool
        Per-sample artifact mask where ``True`` = artifact sample.
        This is the ``artifact_mask`` field from
        ``detect_phase_artifacts``, at the EGG sampling rate.
    trigger_times : array_like
        Scanner trigger onset times in seconds (from
        ``find_scanner_triggers``).
    sfreq : float
        EGG sampling frequency in Hz (e.g., 10.0).
    tr : float
        fMRI repetition time in seconds.
    begin_cut : int, optional
        Volumes to remove from the start. Default is 21.
    end_cut : int, optional
        Volumes to remove from the end. Default is 21.

    Returns
    -------
    vol_mask : np.ndarray of bool
        Per-volume mask where ``True`` = clean volume (include),
        ``False`` = artifact volume (exclude). Length equals
        ``n_volumes - begin_cut - end_cut``.
    """
    sample_mask = np.asarray(sample_mask, dtype=bool)
    trigger_times = np.asarray(trigger_times, dtype=float)
    n_samples = len(sample_mask)
    n_volumes = len(trigger_times)

    vol_clean = np.ones(n_volumes, dtype=bool)
    for k in range(n_volumes):
        t_start = trigger_times[k]
        t_end = trigger_times[k + 1] if k + 1 < n_volumes else t_start + tr
        s = min(int(round(t_start * sfreq)), n_samples)
        e = min(int(round(t_end * sfreq)), n_samples)
        if s < e and np.any(sample_mask[s:e]):
            vol_clean[k] = False

    # Apply volume cuts
    if begin_cut + end_cut >= n_volumes:
        return np.array([], dtype=bool)
    if end_cut == 0:
        return vol_clean[begin_cut:]
    return vol_clean[begin_cut:-end_cut]


def load_bold(bold_path, mask_path):
    """Load fMRIPrep BOLD and brain mask as a masked 2D array.

    Convenience wrapper around nibabel that loads a preprocessed
    BOLD NIfTI and brain mask, applies the mask, and returns the
    data ready for confound regression and phase extraction.

    Parameters
    ----------
    bold_path : str or Path
        Path to the preprocessed BOLD NIfTI file
        (e.g., ``*_desc-preproc_bold.nii.gz``).
    mask_path : str or Path
        Path to the brain mask NIfTI file
        (e.g., ``*_desc-brain_mask.nii.gz``).

    Returns
    -------
    result : dict
        Dictionary with keys:

        - ``bold_2d`` : np.ndarray, shape (n_voxels, n_volumes)
        - ``mask`` : np.ndarray, boolean 3D array
        - ``affine`` : np.ndarray, 4x4 affine matrix
        - ``vol_shape`` : tuple, spatial dimensions (x, y, z)
        - ``n_volumes`` : int
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("nibabel is required: pip install gastropy[neuro]") from e

    bold_img = nib.load(bold_path)
    mask_img = nib.load(mask_path)

    bold_data = bold_img.get_fdata(dtype=np.float32)
    mask = mask_img.get_fdata().astype(bool)

    vol_shape = mask.shape
    n_volumes = bold_data.shape[-1]

    # Apply mask: (x, y, z, t) -> (n_voxels, t)
    bold_2d = bold_data[mask]

    return {
        "bold_2d": bold_2d,
        "mask": mask,
        "affine": bold_img.affine,
        "vol_shape": vol_shape,
        "n_volumes": n_volumes,
    }


def align_bold_to_egg(bold_2d, n_triggers, confounds_df=None):
    """Align BOLD volumes to EGG trigger count.

    fMRIPrep BOLD may contain more volumes than EGG triggers
    (e.g., pre- or post-trigger dummy scans). This function
    truncates BOLD to the first ``n_triggers`` volumes.

    Parameters
    ----------
    bold_2d : np.ndarray, shape (n_voxels, n_volumes)
        BOLD time series.
    n_triggers : int
        Number of EGG scanner triggers (= expected volume count).
    confounds_df : pandas.DataFrame, optional
        If provided, also truncates to first ``n_triggers`` rows.

    Returns
    -------
    bold_aligned : np.ndarray, shape (n_voxels, n_triggers)
        Truncated BOLD time series.
    confounds_aligned : pandas.DataFrame or None
        Truncated confounds (if input was provided).
    """
    n_volumes = bold_2d.shape[1]

    if n_volumes < n_triggers:
        raise ValueError(
            f"BOLD has {n_volumes} volumes but EGG has {n_triggers} "
            f"triggers. BOLD must have at least as many volumes as triggers."
        )

    bold_aligned = bold_2d[:, :n_triggers]

    confounds_aligned = None
    if confounds_df is not None:
        confounds_aligned = confounds_df.iloc[:n_triggers].reset_index(drop=True)

    return bold_aligned, confounds_aligned


def to_nifti(data_3d, affine):
    """Convert a 3D numpy array to a nibabel NIfTI image.

    Parameters
    ----------
    data_3d : np.ndarray
        3D volume data.
    affine : np.ndarray
        4x4 affine transformation matrix.

    Returns
    -------
    img : nibabel.Nifti1Image
        NIfTI image object (can be saved or passed to nilearn).
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("nibabel is required: pip install gastropy[neuro]") from e

    return nib.Nifti1Image(np.asarray(data_3d, dtype=np.float32), affine)


__all__ = [
    "find_scanner_triggers",
    "create_volume_windows",
    "phase_per_volume",
    "apply_volume_cuts",
    "artifact_mask_to_volumes",
    "regress_confounds",
    "bold_voxelwise_phases",
    "compute_plv_map",
    "compute_surrogate_plv_map",
    "load_bold",
    "align_bold_to_egg",
    "to_nifti",
]
