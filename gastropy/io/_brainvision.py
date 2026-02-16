"""BrainVision to BIDS physio converter.

Requires MNE-Python (optional dependency, lazy-imported at call time).
"""

from pathlib import Path

__all__ = ["brainvision_to_bids"]


def brainvision_to_bids(
    vhdr_path,
    output_dir,
    subject,
    task="rest",
    session=None,
    target_sfreq=None,
    channel_pick=None,
):
    """Convert a BrainVision EGG recording to BIDS physio format.

    Reads a BrainVision ``.vhdr`` file via MNE-Python, optionally
    downsamples and selects channels, then writes BIDS-compliant
    ``_physio.tsv.gz`` and ``_physio.json`` files.

    Parameters
    ----------
    vhdr_path : str or Path
        Path to the BrainVision ``.vhdr`` header file.
    output_dir : str or Path
        Root directory for BIDS output.  Files are written to
        ``output_dir/sub-XX/[ses-YY/]beh/``.
    subject : str
        Subject identifier (without ``sub-`` prefix).
    task : str, optional
        Task label for the BIDS filename.  Default ``"rest"``.
    session : str, optional
        Session identifier (without ``ses-`` prefix).  If ``None``,
        the session level is omitted from the output path.
    target_sfreq : float, optional
        Target sampling frequency for downsampling.  If ``None``,
        the original sampling rate is preserved.
    channel_pick : list of str, optional
        Channel names to include.  If ``None``, all channels are kept.

    Returns
    -------
    dict
        Dictionary with keys ``"tsv_path"`` and ``"json_path"``
        pointing to the written files.

    Raises
    ------
    ImportError
        If MNE-Python is not installed.
    """
    try:
        import mne
    except ImportError as err:
        raise ImportError(
            "MNE-Python is required for BrainVision conversion. Install it with: pip install gastropy[neuro]"
        ) from err

    from ._bids import write_bids_physio

    vhdr_path = Path(vhdr_path)
    output_dir = Path(output_dir)

    # Read BrainVision file
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    # Pick channels
    if channel_pick is not None:
        raw.pick(channel_pick)

    # Downsample
    if target_sfreq is not None and target_sfreq < raw.info["sfreq"]:
        raw.resample(target_sfreq, verbose=False)

    sfreq = raw.info["sfreq"]
    signal = raw.get_data()  # (n_channels, n_samples)
    ch_names = raw.ch_names

    # Build BIDS output path
    parts = [f"sub-{subject}"]
    if session is not None:
        parts.append(f"ses-{session}")
    parts.append("beh")
    bids_dir = output_dir.joinpath(*parts)

    # Build BIDS filename
    name_parts = [f"sub-{subject}"]
    if session is not None:
        name_parts.append(f"ses-{session}")
    name_parts.append(f"task-{task}")
    name_parts.append("physio.tsv.gz")
    bids_filename = "_".join(name_parts)

    tsv_path = bids_dir / bids_filename

    tsv_out, json_out = write_bids_physio(
        tsv_path,
        signal=signal,
        sfreq=sfreq,
        columns=list(ch_names),
        start_time=0.0,
        Source=str(vhdr_path.name),
        Description=f"BrainVision EGG converted from {vhdr_path.name}",
    )

    return {"tsv_path": tsv_out, "json_path": json_out}
