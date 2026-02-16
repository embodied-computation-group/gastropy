"""One-time script to convert bundled NPZ sample data to BIDS physio format.

Reads existing .npz files from gastropy/data/ and writes BIDS-compliant
_physio.tsv.gz + _physio.json pairs alongside them.

Usage:
    py scripts/convert_npz_to_bids.py
"""

from pathlib import Path

import numpy as np

from gastropy.io import write_bids_physio

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "gastropy" / "data"


def convert_standalone_egg():
    """Convert egg_standalone.npz -> sub-wolpert_task-rest_physio.*"""
    d = dict(np.load(str(DATA_DIR / "egg_standalone.npz"), allow_pickle=False))

    tsv_path = DATA_DIR / "sub-wolpert_task-rest_physio.tsv.gz"
    columns = list(d["ch_names"])

    write_bids_physio(
        tsv_path,
        signal=d["signal"],
        sfreq=float(d["sfreq"]),
        columns=columns,
        start_time=0.0,
        Source=str(d["source"]),
        Description="7-channel EGG from Wolpert et al. (2020), downsampled to 10 Hz",
        License="CC BY-NC-SA 3.0",
        Citation="Wolpert, N., Rebollo, I., & Tallon-Baudry, C. (2020). Psychophysiology, 57, e13599.",
    )
    print(f"  Wrote {tsv_path.name} ({tsv_path.stat().st_size / 1024:.0f} KB)")


def convert_fmri_session(session_id):
    """Convert fmri_egg_session_XXXX.npz -> sub-01_ses-XXXX_task-rest_physio.*"""
    d = dict(np.load(str(DATA_DIR / f"fmri_egg_session_{session_id}.npz"), allow_pickle=False))

    signal = d["signal"]  # (8, n_samples)
    sfreq = float(d["sfreq"])  # 10.0
    trigger_times = d["trigger_times"]

    # Build trigger column (1 at nearest sample to each trigger, 0 elsewhere)
    n_samples = signal.shape[1]
    trigger_col = np.zeros((1, n_samples), dtype=np.float64)
    trigger_indices = np.round(trigger_times * sfreq).astype(int)
    trigger_indices = trigger_indices[trigger_indices < n_samples]
    trigger_col[0, trigger_indices] = 1.0

    # Stack: (9, n_samples) = 8 EGG + 1 trigger
    full_signal = np.vstack([signal, trigger_col])
    columns = list(d["ch_names"]) + ["trigger"]

    tsv_path = DATA_DIR / f"sub-01_ses-{session_id}_task-rest_physio.tsv.gz"

    write_bids_physio(
        tsv_path,
        signal=full_signal,
        sfreq=sfreq,
        columns=columns,
        start_time=0.0,
        TR=float(d["tr"]),
        Source=str(d["source"]),
        Session=str(d["session"]),
        TriggerTimesSeconds=trigger_times.tolist(),
        Description=f"8-channel EGG during fMRI session {session_id}, downsampled to 10 Hz",
    )
    print(f"  Wrote {tsv_path.name} ({tsv_path.stat().st_size / 1024:.0f} KB)")


def verify_round_trip():
    """Verify BIDS files match original NPZ data."""
    from gastropy.io import read_bids_physio

    print("\n=== Verifying round-trip fidelity ===")

    # Standalone EGG
    npz = dict(np.load(str(DATA_DIR / "egg_standalone.npz"), allow_pickle=False))
    bids = read_bids_physio(DATA_DIR / "sub-wolpert_task-rest_physio.tsv.gz")
    np.testing.assert_allclose(bids["signal"], npz["signal"], atol=1e-10)
    print("  egg_standalone: OK")

    # fMRI-EGG sessions
    for sid in ("0001", "0003", "0004", "0008"):
        npz = dict(np.load(str(DATA_DIR / f"fmri_egg_session_{sid}.npz"), allow_pickle=False))
        bids = read_bids_physio(DATA_DIR / f"sub-01_ses-{sid}_task-rest_physio.tsv.gz")

        # EGG channels (first 8 rows)
        np.testing.assert_allclose(bids["signal"][:8], npz["signal"], atol=1e-10)

        # Trigger times from JSON sidecar
        np.testing.assert_allclose(bids["TriggerTimesSeconds"], npz["trigger_times"].tolist(), atol=1e-12)

        print(f"  fmri_egg_session_{sid}: OK")


def main():
    print("=== Converting standalone EGG ===")
    convert_standalone_egg()

    print("\n=== Converting fMRI-EGG sessions ===")
    for sid in ("0001", "0003", "0004", "0008"):
        convert_fmri_session(sid)

    verify_round_trip()

    print("\nDone. Run: py -m pytest tests/test_data.py -v")
    print("Then delete the .npz files.")


if __name__ == "__main__":
    main()
