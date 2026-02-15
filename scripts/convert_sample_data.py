"""One-time script to convert source EGG data into compact .npz sample files.

Sources:
  - Semi_precision FIF files (fMRI-EGG, 8 channels, 1000 Hz)
  - Wolpert EGG_Scripts MAT/HDF5 files (standalone EGG, 7 channels, 1000 Hz)

Output:
  - gastropy/data/fmri_egg_session_XXXX.npz  (downsampled to 10 Hz)
  - gastropy/data/egg_standalone.npz          (downsampled to 10 Hz)
"""

from pathlib import Path

import h5py
import mne
import numpy as np
from scipy.signal import resample

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "gastropy" / "data"

SEMI_PRECISION_DIR = Path(
    r"c:\Users\Micah\vibes\semi_precision\derivatives\egg\segments"
    r"\corrected_extraction_all_sessions_20250912_225310"
)

WOLPERT_DIR = Path(r"c:\Users\Micah\vibes\EGG_Scripts")

# --- Config ---
TARGET_SFREQ = 10.0  # Hz — standard for gastropy
SEMI_PRECISION_SESSIONS = ["0001", "0003", "0004", "0008"]
TR = 1.856  # seconds


def convert_fif_session(session_id: str) -> None:
    """Convert one semi_precision FIF session to .npz."""
    fif_path = SEMI_PRECISION_DIR / f"semi_precision_{session_id}_firstrest_420vols.fif"
    if not fif_path.exists():
        print(f"  SKIP {session_id}: {fif_path} not found")
        return

    print(f"  Loading {fif_path.name} ...")
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

    # Extract data
    data = raw.get_data()  # (n_channels, n_samples) in Volts
    orig_sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    duration_s = raw.times[-1]

    # Extract R128 trigger times
    trigger_times = np.array([ann["onset"] for ann in raw.annotations if "R128" in ann["description"]])

    # Downsample
    n_target = int(len(raw.times) * TARGET_SFREQ / orig_sfreq)
    data_resampled = resample(data, n_target, axis=1)

    # Save
    out_path = OUTPUT_DIR / f"fmri_egg_session_{session_id}.npz"
    np.savez_compressed(
        out_path,
        signal=data_resampled.astype(np.float64),
        sfreq=np.float64(TARGET_SFREQ),
        ch_names=np.array(ch_names),
        trigger_times=trigger_times.astype(np.float64),
        tr=np.float64(TR),
        duration_s=np.float64(duration_s),
        source=np.array("semi_precision"),
        session=np.array(session_id),
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved {out_path.name}: {data_resampled.shape}, {size_kb:.0f} KB")


def convert_wolpert_mat() -> None:
    """Convert Wolpert example2.mat to .npz."""
    mat_path = WOLPERT_DIR / "EGG_raw_example2.mat"
    if not mat_path.exists():
        print(f"  SKIP: {mat_path} not found")
        return

    print(f"  Loading {mat_path.name} (HDF5) ...")
    with h5py.File(str(mat_path), "r") as f:
        g = f["EGG_raw"]

        # Sampling rate
        orig_sfreq = g["fsample"][()].flat[0]

        # Channel labels
        label_ds = g["label"]
        ch_names = []
        for idx in np.ndindex(label_ds.shape):
            ref = label_ds[idx]
            chars = f[ref][()]
            ch_names.append("".join(chr(c) for c in chars.flat))

        # Data — must transpose (MATLAB column-major → numpy row-major)
        trial_ref = g["trial"][0, 0]
        data = f[trial_ref][()].T  # (n_channels, n_samples)

    # Downsample
    n_samples = data.shape[1]
    duration_s = n_samples / orig_sfreq
    n_target = int(n_samples * TARGET_SFREQ / orig_sfreq)
    data_resampled = resample(data, n_target, axis=1)

    # Save
    out_path = OUTPUT_DIR / "egg_standalone.npz"
    np.savez_compressed(
        out_path,
        signal=data_resampled.astype(np.float64),
        sfreq=np.float64(TARGET_SFREQ),
        ch_names=np.array(ch_names),
        duration_s=np.float64(duration_s),
        source=np.array("wolpert_2020"),
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"  Saved {out_path.name}: {data_resampled.shape}, {size_kb:.0f} KB")


def main():
    print("=== Converting semi_precision fMRI-EGG sessions ===")
    for session_id in SEMI_PRECISION_SESSIONS:
        convert_fif_session(session_id)

    print("\n=== Converting Wolpert standalone EGG ===")
    convert_wolpert_mat()

    print("\nDone. Files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
