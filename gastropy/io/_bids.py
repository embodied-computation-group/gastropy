"""BIDS physio file reading and writing.

Implements the BIDS peripheral physiology format:
- ``_physio.tsv.gz`` — gzip-compressed TSV, no header, one column per channel
- ``_physio.json`` — sidecar with SamplingFrequency, StartTime, Columns

Only uses stdlib (``gzip``, ``json``) and numpy — no pybids dependency.
"""

import gzip
import json
from pathlib import Path

import numpy as np

__all__ = ["read_bids_physio", "write_bids_physio", "parse_bids_filename"]


def read_bids_physio(tsv_path, json_path=None):
    """Read a BIDS ``_physio.tsv.gz`` and its companion JSON sidecar.

    Parameters
    ----------
    tsv_path : str or Path
        Path to a ``_physio.tsv.gz`` (or uncompressed ``_physio.tsv``) file.
        Gzip compression is auto-detected from the ``.gz`` extension.
    json_path : str or Path, optional
        Path to the companion ``_physio.json`` sidecar.  If ``None``,
        inferred by replacing the TSV extension with ``.json``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``signal`` : ndarray, shape ``(n_channels, n_samples)``
        - ``sfreq`` : float — sampling frequency in Hz
        - ``columns`` : list of str — column names from the JSON ``Columns``
          field

        Plus every other key-value pair from the JSON sidecar (e.g.
        ``StartTime``, ``TR``, ``Source``) stored as-is.

    Raises
    ------
    FileNotFoundError
        If the TSV or JSON file does not exist.
    ValueError
        If the JSON sidecar is missing required BIDS fields
        (``SamplingFrequency``, ``Columns``).
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    # Resolve JSON sidecar path
    json_path = _tsv_to_json_path(tsv_path) if json_path is None else Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON sidecar not found: {json_path}")

    # Read JSON sidecar
    with open(json_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate required BIDS fields
    for field in ("SamplingFrequency", "Columns"):
        if field not in metadata:
            raise ValueError(f"JSON sidecar missing required BIDS field: {field}")

    # Read TSV (gzipped or plain)
    if tsv_path.suffix == ".gz":
        with gzip.open(tsv_path, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        with open(tsv_path, encoding="utf-8") as f:
            text = f.read()

    # Parse TSV → numpy array
    # BIDS physio TSV: no header, tab-separated, one row per sample
    lines = text.strip().split("\n")
    n_samples = len(lines)
    n_columns = len(metadata["Columns"])

    data = np.empty((n_samples, n_columns), dtype=np.float64)
    for i, line in enumerate(lines):
        data[i] = [float(v) for v in line.split("\t")]

    # Transpose to (n_channels, n_samples) — GastroPy convention
    signal = data.T

    result = {
        "signal": signal,
        "sfreq": float(metadata["SamplingFrequency"]),
        "columns": list(metadata["Columns"]),
    }

    # Include all other JSON fields
    for key, value in metadata.items():
        if key not in ("SamplingFrequency", "Columns"):
            result[key] = value

    return result


def write_bids_physio(tsv_path, signal, sfreq, columns, start_time=0.0, **extra_json):
    """Write signal data as BIDS ``_physio.tsv.gz`` with companion JSON sidecar.

    Parameters
    ----------
    tsv_path : str or Path
        Output path for the ``.tsv.gz`` file.  The companion ``.json``
        sidecar is written alongside with the same stem.
    signal : ndarray, shape ``(n_channels, n_samples)``
        Signal data.  Each row becomes one column in the TSV.
    sfreq : float
        Sampling frequency in Hz.
    columns : list of str
        Column names, one per row of *signal*.  Length must match
        ``signal.shape[0]``.
    start_time : float, optional
        Recording start time in seconds relative to task onset.
        Default is ``0.0``.
    **extra_json
        Additional key-value pairs for the JSON sidecar (e.g.
        ``TR=1.856``, ``Source="semi_precision"``).

    Returns
    -------
    tuple of (Path, Path)
        Paths to the written ``(tsv_path, json_path)``.

    Raises
    ------
    ValueError
        If ``len(columns)`` does not match ``signal.shape[0]``.
    """
    tsv_path = Path(tsv_path)
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    if len(columns) != signal.shape[0]:
        raise ValueError(f"Number of columns ({len(columns)}) does not match signal rows ({signal.shape[0]})")

    # Ensure parent directory exists
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write gzipped TSV — no header row per BIDS spec
    # Transpose: (n_channels, n_samples) → (n_samples, n_channels)
    data = signal.T
    with gzip.open(tsv_path, "wt", encoding="utf-8") as f:
        for row in data:
            f.write("\t".join(f"{v:.10g}" for v in row) + "\n")

    # Write JSON sidecar
    json_path = _tsv_to_json_path(tsv_path)
    metadata = {
        "SamplingFrequency": float(sfreq),
        "StartTime": float(start_time),
        "Columns": list(columns),
    }
    metadata.update(extra_json)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        f.write("\n")

    return tsv_path, json_path


def parse_bids_filename(path):
    """Parse BIDS key-value entities from a filename.

    Parameters
    ----------
    path : str or Path
        A BIDS-style filename, e.g.
        ``sub-01_ses-0001_task-rest_physio.tsv.gz``.

    Returns
    -------
    dict
        Dictionary of BIDS entities.  Always includes ``"suffix"``
        (the final label before the extension) and ``"extension"``.

        Example::

            {"sub": "01", "ses": "0001", "task": "rest",
             "suffix": "physio", "extension": ".tsv.gz"}
    """
    path = Path(path)
    name = path.name

    # Handle compound extensions like .tsv.gz
    if name.endswith(".tsv.gz"):
        extension = ".tsv.gz"
        stem = name[: -len(".tsv.gz")]
    else:
        extension = path.suffix
        stem = path.stem

    # Split on underscores, last part is the suffix
    parts = stem.split("_")
    suffix = parts[-1] if parts else ""
    entity_parts = parts[:-1]

    entities = {}
    for part in entity_parts:
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value

    entities["suffix"] = suffix
    entities["extension"] = extension

    return entities


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _tsv_to_json_path(tsv_path):
    """Derive JSON sidecar path from a TSV path."""
    tsv_path = Path(tsv_path)
    name = tsv_path.name
    if name.endswith(".tsv.gz"):
        json_name = name[: -len(".tsv.gz")] + ".json"
    elif name.endswith(".tsv"):
        json_name = name[: -len(".tsv")] + ".json"
    else:
        json_name = tsv_path.stem + ".json"
    return tsv_path.parent / json_name
