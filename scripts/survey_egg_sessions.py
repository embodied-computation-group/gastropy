"""Survey all fMRI-concurrent EGG sessions to compare quality.

Follows the fMRI-EGG pipeline from semi_precision:
  1. Load .fif (8ch, 1000 Hz) → resample to 10 Hz
  2. Select best channel via PSD on unfiltered 10 Hz data
  3. Bandpass filter (FIR) at 10 Hz
  4. Resample filtered signal to fMRI rate (1/TR Hz)
  5. Hilbert transform on fMRI-rate signal
  6. Create volume windows from R128 triggers
  7. Mean phase per volume → apply edge cuts (21 volumes)
  8. Quality metrics + artifact detection + plots

Produces per-session plots, a comparison figure, and a ranked CSV summary.

Usage:
    py scripts/survey_egg_sessions.py
"""

import sys
import traceback
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Ensure gastropy is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gastropy.egg as egg
import gastropy.metrics as metrics
import gastropy.viz as viz
from gastropy.neuro.fmri import (
    apply_volume_cuts,
    create_volume_windows,
    find_scanner_triggers,
    phase_per_volume,
)
from gastropy.signal import (
    detect_phase_artifacts,
    instantaneous_phase,
    psd_welch,
    resample_signal,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EGG_DIR = Path(
    r"c:\Users\Micah\vibes\semi_precision\derivatives\egg\segments"
    r"\corrected_extraction_all_sessions_20250912_225310"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "egg_survey"

WORK_SFREQ = 10.0  # intermediate working rate (Hz)
TR = 1.856  # fMRI repetition time (seconds)
FMRI_SFREQ = 1.0 / TR  # ~0.539 Hz
N_VOLUMES = 420  # expected volumes per session
BEGIN_CUT = 21  # volumes to trim from start
END_CUT = 21  # volumes to trim from end


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_and_resample(fif_path):
    """Load .fif, extract R128 triggers, resample EGG to WORK_SFREQ.

    Returns (data_10hz, ch_names, trigger_onsets_s, orig_sfreq).
    """
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    orig_sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names

    # Extract R128 scanner triggers before resampling
    trigger_onsets = find_scanner_triggers(raw.annotations, label="R128")

    # Resample each channel to 10 Hz
    data_orig = raw.get_data()  # (n_ch, n_samples)
    data_10hz = np.zeros((data_orig.shape[0], int(round(data_orig.shape[1] * WORK_SFREQ / orig_sfreq))))
    for ch_idx in range(data_orig.shape[0]):
        data_10hz[ch_idx], _ = resample_signal(data_orig[ch_idx], orig_sfreq, WORK_SFREQ)

    return data_10hz, ch_names, trigger_onsets, orig_sfreq


def process_one_session(session_id, fif_path, out_dir):
    """Run full fMRI-EGG pipeline on one session.

    Returns a dict of summary metrics, or None on failure.
    """
    print(f"\n{'=' * 60}")
    print(f"Session {session_id}")
    print(f"{'=' * 60}")

    # Step 1: Load + resample to 10 Hz
    data, ch_names, triggers, orig_sfreq = load_and_resample(fif_path)
    n_ch, n_samp = data.shape
    duration_s = n_samp / WORK_SFREQ
    print(f"  {n_ch} ch, {orig_sfreq:.0f} Hz -> {n_samp} samples @ {WORK_SFREQ} Hz, {duration_s:.1f} s")
    print(f"  R128 triggers: {len(triggers)}")

    # Step 2: Channel selection on unfiltered 10 Hz data
    best_idx, peak_freq, freqs_best, psd_best = egg.select_best_channel(data, WORK_SFREQ)
    best_ch = ch_names[best_idx]
    print(f"  Best channel: {best_ch} (idx {best_idx}), peak {peak_freq * 60:.2f} cpm")

    # Compute multi-channel PSD for plotting
    psd_all = []
    for ch_idx in range(n_ch):
        freqs, psd_ch = psd_welch(data[ch_idx], WORK_SFREQ, fmin=0.0, fmax=0.1)
        psd_all.append(psd_ch)
    psd_all = np.array(psd_all)  # (n_ch, n_freqs)

    # Step 3: Bandpass filter at 10 Hz (via egg_process)
    signals, info = egg.egg_process(data[best_idx], WORK_SFREQ)

    # Step 4: Resample filtered signal to fMRI rate
    filtered_10hz = signals["filtered"].values
    filtered_fmri, actual_fmri_sfreq = resample_signal(filtered_10hz, WORK_SFREQ, FMRI_SFREQ)
    print(f"  Resampled to fMRI rate: {len(filtered_fmri)} samples @ {actual_fmri_sfreq:.4f} Hz")

    # Step 5: Hilbert on fMRI-rate signal
    phase_fmri, analytic_fmri = instantaneous_phase(filtered_fmri)

    # Step 6: Volume windows from R128 triggers
    n_vols = min(N_VOLUMES, len(triggers), len(filtered_fmri))
    windows = create_volume_windows(triggers, TR, n_vols)
    vol_phases = phase_per_volume(analytic_fmri, windows)
    print(f"  Volume phases: {len(vol_phases)} volumes")

    # Step 7: Edge trimming
    vol_phases_cut = apply_volume_cuts(vol_phases, BEGIN_CUT, END_CUT)
    print(f"  After cuts ({BEGIN_CUT}+{END_CUT}): {len(vol_phases_cut)} volumes")

    # Step 8: Quality metrics (from 10 Hz continuous processing)
    cs = info["cycle_stats"]
    quality = metrics.assess_quality(cs["n_cycles"], cs["sd_cycle_dur_s"], info["proportion_normogastric"])

    # Artifact detection on continuous 10 Hz phase
    times_10hz = np.arange(len(signals)) / WORK_SFREQ
    artifact_info = detect_phase_artifacts(signals["phase"].values, times_10hz)

    label = "PASS" if quality["overall"] else "FAIL"
    print(f"  Cycles: {cs['n_cycles']}, duration {cs['mean_cycle_dur_s']:.1f} +/- {cs['sd_cycle_dur_s']:.1f} s")
    print(f"  IC: {info['instability_coefficient']:.3f}")
    print(f"  Normogastric: {info['proportion_normogastric']:.0%}")
    print(f"  Band power: {info['band_power']['prop_power']:.3f}")
    print(f"  Artifacts: {artifact_info['n_artifacts']}")
    print(f"  Quality: {label}")

    # --- Plots ---
    sdir = out_dir / session_id
    sdir.mkdir(parents=True, exist_ok=True)

    # PSD (all channels, unfiltered)
    fig, ax = viz.plot_psd(freqs, psd_all, ch_names=ch_names, best_idx=best_idx, peak_freq=peak_freq)
    fig.suptitle(f"Session {session_id} — PSD (unfiltered, 10 Hz)", fontsize=12)
    fig.savefig(sdir / "psd.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # EGG overview (10 Hz: raw, filtered, phase, amplitude)
    fig, _ = viz.plot_egg_overview(signals, WORK_SFREQ, title=f"Session {session_id} — {best_ch} (10 Hz)")
    fig.savefig(sdir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Cycle histogram
    fig, ax = viz.plot_cycle_histogram(info["cycle_durations_s"])
    ax.set_title(f"Session {session_id} — Cycle durations")
    fig.savefig(sdir / "cycles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Phase artifacts (10 Hz)
    fig, ax = viz.plot_artifacts(signals["phase"].values, times_10hz, artifact_info)
    ax.set_title(f"Session {session_id} — Phase artifacts ({artifact_info['n_artifacts']})")
    fig.savefig(sdir / "artifacts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Volume-level phase (fMRI rate, with cuts)
    fig, ax = viz.plot_volume_phase(vol_phases, tr=TR, cut_start=BEGIN_CUT, cut_end=END_CUT)
    ax.set_title(f"Session {session_id} — Phase per volume (TR={TR}s)")
    fig.savefig(sdir / "volume_phase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "session": session_id,
        "best_channel": best_ch,
        "peak_freq_cpm": round(info["peak_freq_hz"] * 60, 2),
        "n_cycles": cs["n_cycles"],
        "mean_cycle_s": round(cs["mean_cycle_dur_s"], 1),
        "sd_cycle_s": round(cs["sd_cycle_dur_s"], 1),
        "instability": round(info["instability_coefficient"], 3),
        "prop_normo": round(info["proportion_normogastric"], 3),
        "band_power": round(info["band_power"]["prop_power"], 3),
        "n_artifacts": artifact_info["n_artifacts"],
        "n_triggers": len(triggers),
        "n_vol_phases": len(vol_phases_cut),
        "quality": label,
        "duration_s": round(duration_s, 1),
    }


def make_comparison_figure(df, out_dir):
    """Bar-chart comparison across sessions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sessions = df["session"].values

    ax = axes[0, 0]
    colors = ["tab:green" if v >= 0.7 else "tab:red" for v in df["prop_normo"]]
    ax.barh(sessions, df["prop_normo"], color=colors)
    ax.axvline(0.7, color="grey", ls="--", alpha=0.6, label="70% threshold")
    ax.set_xlabel("Proportion normogastric")
    ax.set_title("Normogastric proportion")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.barh(sessions, df["band_power"], color="steelblue")
    ax.set_xlabel("Proportion of total power")
    ax.set_title("Band power (2-4 cpm)")

    ax = axes[0, 2]
    ax.barh(sessions, df["instability"], color="coral")
    ax.set_xlabel("Instability coefficient")
    ax.set_title("Rhythm instability (lower = better)")

    ax = axes[1, 0]
    ax.barh(sessions, df["peak_freq_cpm"], color="mediumpurple")
    ax.axvline(2, color="grey", ls="--", alpha=0.4)
    ax.axvline(4, color="grey", ls="--", alpha=0.4)
    ax.set_xlabel("Cycles per minute")
    ax.set_title("Peak frequency")

    ax = axes[1, 1]
    ax.barh(sessions, df["n_cycles"], color="teal")
    ax.axvline(10, color="grey", ls="--", alpha=0.4, label="min 10 cycles")
    ax.set_xlabel("Number of cycles")
    ax.set_title("Detected cycles")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.barh(sessions, df["n_artifacts"], color="salmon")
    ax.set_xlabel("Count")
    ax.set_title("Phase artifacts")

    fig.suptitle("EGG Quality Comparison — All Sessions (fMRI-EGG pipeline)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    fif_files = sorted(EGG_DIR.glob("*.fif"))
    if not fif_files:
        print(f"No .fif files found in {EGG_DIR}")
        sys.exit(1)

    print(f"Found {len(fif_files)} sessions:")
    for f in fif_files:
        print(f"  {f.name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for fif_path in fif_files:
        # filename: semi_precision_XXXX_firstrest_420vols.fif
        session_id = fif_path.stem.split("_")[2]
        try:
            row = process_one_session(session_id, fif_path, OUTPUT_DIR)
            if row:
                results.append(row)
        except Exception:
            traceback.print_exc()

    if not results:
        print("No sessions processed successfully.")
        sys.exit(1)

    df = pd.DataFrame(results)

    # Rank by composite score (higher = better)
    max_artifacts = max(df["n_artifacts"].max(), 1)
    df["rank_score"] = (
        df["prop_normo"] * 0.4
        + df["band_power"] * 0.3
        + (1 - df["instability"].clip(upper=1)) * 0.2
        + (1 - df["n_artifacts"] / max_artifacts) * 0.1
    )
    df = df.sort_values("rank_score", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 80}")
    print("SUMMARY — All Sessions (ranked by quality)")
    print(f"{'=' * 80}")
    summary_cols = [
        "session",
        "best_channel",
        "peak_freq_cpm",
        "n_cycles",
        "mean_cycle_s",
        "sd_cycle_s",
        "instability",
        "prop_normo",
        "band_power",
        "n_artifacts",
        "n_triggers",
        "quality",
        "rank_score",
    ]
    print(df.to_string(index=False, columns=summary_cols, float_format="%.3f"))

    best = df.iloc[0]
    print(f"\nBest session: {best['session']} (score {best['rank_score']:.3f})")

    # Save outputs
    df.to_csv(OUTPUT_DIR / "session_summary.csv", index=False)
    make_comparison_figure(df, OUTPUT_DIR)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("  session_summary.csv  — ranked metrics table")
    print("  comparison.png       — cross-session bar charts")
    print("  <session>/           — per-session plots (psd, overview, cycles, artifacts, volume_phase)")


if __name__ == "__main__":
    main()
