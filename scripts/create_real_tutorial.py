"""Create the real-data fMRI coupling tutorial notebook."""

import json

cells = []


def md(source):
    lines = source.strip().split("\n")
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
        }
    )


def code(source):
    lines = source.strip().split("\n")
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
        }
    )


# === Cell 0: Title ===
md(
    "# Gastric-Brain Coupling with Real fMRI Data\n"
    "\n"
    "This tutorial demonstrates the complete gastric-brain phase coupling\n"
    "pipeline using real fMRIPrep-preprocessed BOLD data and concurrent EGG.\n"
    "\n"
    "**What you'll learn:**\n"
    "- Loading and aligning fMRI-concurrent EGG with BOLD data\n"
    "- EGG channel selection and narrowband filtering\n"
    "- Spatial smoothing of BOLD data\n"
    "- BOLD confound regression and phase extraction\n"
    "- Artifact detection and volume censoring\n"
    "- Computing voxelwise PLV maps\n"
    "- Surrogate statistical testing\n"
    "- Visualizing volumetric coupling maps with nilearn\n"
    "\n"
    "**Prerequisites:** ``pip install gastropy[neuro]`` (adds nibabel, nilearn, pooch)\n"
    "\n"
    "**Data:** ~1.2 GB download on first run (cached for subsequent runs).\n"
    "Session 0008 from the semi_precision study: 8-channel EGG at 10 Hz,\n"
    "fMRIPrep BOLD in MNI152NLin2009cAsym space (2 mm), TR = 1.856 s.\n"
    "\n"
    "**Expected runtime:** ~10 minutes (mostly surrogate computation)."
)

# === Cell 1: Imports ===
code(
    "import time\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import nibabel as nib\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from nilearn.image import smooth_img\n"
    "\n"
    "import gastropy as gp\n"
    "from gastropy.neuro.fmri import (\n"
    "    align_bold_to_egg,\n"
    "    apply_volume_cuts,\n"
    "    artifact_mask_to_volumes,\n"
    "    bold_voxelwise_phases,\n"
    "    compute_plv_map,\n"
    "    compute_surrogate_plv_map,\n"
    "    create_volume_windows,\n"
    "    phase_per_volume,\n"
    "    regress_confounds,\n"
    "    to_nifti,\n"
    ")\n"
    "from gastropy.signal import detect_phase_artifacts\n"
    "\n"
    'plt.rcParams["figure.dpi"] = 100\n'
    'plt.rcParams["figure.facecolor"] = "white"'
)

# === Cell 2: Section 1 ===
md(
    "## 1. Load Data\n"
    "\n"
    "GastroPy provides ``fetch_fmri_bold`` to download preprocessed BOLD,\n"
    "brain mask, and confounds from a GitHub Release. The EGG data is\n"
    "bundled with the package."
)

# === Cell 3: Load EGG ===
code(
    "# Download BOLD data (~1.2 GB, cached after first run)\n"
    'fmri_paths = gp.fetch_fmri_bold(session="0008")\n'
    "\n"
    "# Load bundled EGG data\n"
    'egg = gp.load_fmri_egg(session="0008")\n'
    "\n"
    'print("EGG data:")\n'
    "print(f\"  Signal shape: {egg['signal'].shape} (channels x samples)\")\n"
    "print(f\"  Sampling rate: {egg['sfreq']} Hz\")\n"
    "print(f\"  Scanner triggers: {len(egg['trigger_times'])}\")\n"
    "print(f\"  TR: {egg['tr']} s\")\n"
    "print(f\"  Duration: {egg['duration_s']:.0f} s ({egg['duration_s']/60:.1f} min)\")"
)

# === Cell 4: Load and smooth BOLD ===
md(
    "### 1b. Load and Smooth BOLD\n"
    "\n"
    "We load the BOLD NIfTI directly with nibabel, apply 6 mm FWHM Gaussian\n"
    "spatial smoothing with nilearn, then mask to extract brain voxels.\n"
    "Smoothing before masking is standard practice."
)

# === Cell 5: Load BOLD with smoothing ===
code(
    "t0 = time.time()\n"
    "\n"
    "# Load NIfTI images\n"
    'bold_img = nib.load(fmri_paths["bold"])\n'
    'mask_img = nib.load(fmri_paths["mask"])\n'
    "\n"
    "# Spatial smoothing (6 mm FWHM Gaussian)\n"
    "bold_smooth = smooth_img(bold_img, fwhm=6)\n"
    "\n"
    "# Extract brain voxels using the mask\n"
    "mask_data = mask_img.get_fdata().astype(bool)\n"
    "bold_4d = bold_smooth.get_fdata(dtype=np.float32)\n"
    "vol_shape = mask_data.shape\n"
    "n_volumes = bold_4d.shape[-1]\n"
    "bold_2d_all = bold_4d[mask_data]  # (n_voxels, n_volumes)\n"
    "affine = bold_img.affine\n"
    "\n"
    "elapsed = time.time() - t0\n"
    'print(f"BOLD loaded + smoothed (6 mm) in {elapsed:.1f} s")\n'
    'print(f"  Volume shape: {vol_shape}")\n'
    'print(f"  Volumes: {n_volumes}")\n'
    'print(f"  Brain voxels: {bold_2d_all.shape[0]:,}")\n'
    'print(f"  Memory: {bold_2d_all.nbytes / 1e9:.2f} GB")\n'
    "\n"
    "# Load confounds\n"
    'confounds = pd.read_csv(fmri_paths["confounds"], sep="\\t")\n'
    'print(f"\\nConfounds: {confounds.shape[0]} rows x {confounds.shape[1]} columns")'
)

# === Cell 6: Align section ===
md(
    "## 2. Align BOLD Volumes to EGG Triggers\n"
    "\n"
    "The BOLD file from fMRIPrep may contain more volumes than EGG triggers\n"
    "(e.g., dummy scans at the start/end). We align by keeping only the\n"
    "BOLD volumes that correspond to EGG scanner triggers."
)

# === Cell 7: Align ===
code(
    'n_triggers = len(egg["trigger_times"])\n'
    'print(f"BOLD volumes: {n_volumes}")\n'
    'print(f"EGG triggers: {n_triggers}")\n'
    'print(f"Discarding {n_volumes - n_triggers} extra BOLD volumes")\n'
    "\n"
    "bold_2d, confounds_aligned = align_bold_to_egg(\n"
    "    bold_2d_all, n_triggers, confounds\n"
    ")\n"
    'print(f"\\nAligned BOLD: {bold_2d.shape}")\n'
    'print(f"Aligned confounds: {confounds_aligned.shape}")'
)

# === Cell 8: EGG section ===
md(
    "## 3. EGG Processing: Channel Selection and Filtering\n"
    "\n"
    "We identify the best EGG channel (strongest gastric rhythm) and its\n"
    "individual peak frequency, then narrowband filter the EGG at that\n"
    "frequency."
)

# === Cell 9: EGG processing ===
code(
    'sfreq = egg["sfreq"]\n'
    'ch_names = list(egg["ch_names"])\n'
    "\n"
    "# Select best channel and peak frequency\n"
    'best_idx, peak_freq, freqs, psd = gp.select_best_channel(egg["signal"], sfreq)\n'
    'print(f"Best channel: {ch_names[best_idx]} (index {best_idx})")\n'
    'print(f"Peak frequency: {peak_freq:.4f} Hz ({peak_freq * 60:.2f} cpm)")\n'
    "\n"
    "# Compute all-channel PSD for plotting\n"
    "all_psd = np.column_stack(\n"
    '    [gp.psd_welch(egg["signal"][i], sfreq)[1] for i in range(egg["signal"].shape[0])]\n'
    ").T"
)

# === Cell 10: PSD plot ===
code(
    "fig, ax = gp.plot_psd(freqs, all_psd, best_idx=best_idx, peak_freq=peak_freq, ch_names=ch_names)\n"
    'ax.set_title(f"EGG Power Spectral Density \\u2014 Best: {ch_names[best_idx]}")\n'
    "plt.show()"
)

# === Cell 11: Filter ===
code(
    "# Narrowband filter at individual peak frequency\n"
    "hwhm = 0.015  # Hz (half-width at half-maximum)\n"
    "low_hz = peak_freq - hwhm\n"
    "high_hz = peak_freq + hwhm\n"
    'print(f"Filter band: {low_hz:.4f} - {high_hz:.4f} Hz")\n'
    "\n"
    "filtered, filt_info = gp.apply_bandpass(\n"
    '    egg["signal"][best_idx], sfreq, low_hz=low_hz, high_hz=high_hz\n'
    ")\n"
    "phase, analytic = gp.instantaneous_phase(filtered)\n"
    "print(f\"Filter taps: {filt_info.get('fir_numtaps', 'N/A')}\")"
)

# === Cell 12: Phase section ===
md(
    "## 4. Per-Volume EGG Phase\n"
    "\n"
    "Map the continuous EGG phase (10 Hz) to one phase value per fMRI volume,\n"
    "then trim 21 transient volumes from each edge (standard practice to\n"
    "remove filter ringing artifacts)."
)

# === Cell 13: Phase per volume ===
code(
    'windows = create_volume_windows(egg["trigger_times"], egg["tr"], n_triggers)\n'
    "egg_vol_phase = phase_per_volume(analytic, windows)\n"
    "\n"
    "begin_cut, end_cut = 21, 21\n"
    "egg_phase = apply_volume_cuts(egg_vol_phase, begin_cut, end_cut)\n"
    "\n"
    'print(f"Per-volume phases: {len(egg_vol_phase)}")\n'
    'print(f"After trimming ({begin_cut} + {end_cut}): {len(egg_phase)}")'
)

# === Cell 14: Phase plot ===
code(
    'fig, ax = gp.plot_volume_phase(egg_vol_phase, tr=egg["tr"], cut_start=begin_cut, cut_end=end_cut)\n'
    'ax.set_title("EGG Phase Per Volume (shaded = trimmed transients)")\n'
    "plt.show()"
)

# === Cell 15: Artifact section ===
md(
    "## 5. Artifact Detection and Volume Censoring\n"
    "\n"
    "Detect phase artifacts in the continuous 10 Hz EGG phase and map them\n"
    "to volume-level. Volumes containing any artifact sample are censored\n"
    "(excluded from PLV computation)."
)

# === Cell 16: Artifact detection ===
code(
    "# Detect phase artifacts on continuous 10 Hz signal\n"
    "times_10hz = np.arange(len(phase)) / sfreq\n"
    "artifact_info = detect_phase_artifacts(phase, times_10hz)\n"
    "\n"
    "print(f\"Artifacts detected: {artifact_info['n_artifacts']}\")\n"
    "print(f\"Artifact samples: {artifact_info['artifact_mask'].sum()} / {len(phase)}\")\n"
    "\n"
    "# Map sample-level artifacts to volume-level mask\n"
    "vol_mask = artifact_mask_to_volumes(\n"
    "    artifact_info['artifact_mask'],\n"
    "    egg['trigger_times'],\n"
    "    sfreq,\n"
    "    egg['tr'],\n"
    "    begin_cut=begin_cut,\n"
    "    end_cut=end_cut,\n"
    ")\n"
    "\n"
    'print(f"\\nVolume mask: {len(vol_mask)} volumes")\n'
    'print(f"  Clean: {vol_mask.sum()}")\n'
    'print(f"  Censored: {(~vol_mask).sum()}")'
)

# === Cell 17: Artifact plot ===
code(
    "fig, ax = gp.plot_artifacts(phase, times_10hz, artifact_info)\n"
    "ax.set_title(f\"Phase Artifacts ({artifact_info['n_artifacts']} detected)\")\n"
    "plt.show()"
)

# === Cell 18: BOLD section ===
md(
    "## 6. BOLD Processing\n"
    "\n"
    "### 6a. Confound Regression\n"
    "\n"
    "Remove motion and noise confounds from BOLD data using GLM regression.\n"
    "Default regressors: 6 motion parameters + 6 aCompCor components\n"
    "(12 total)."
)

# === Cell 19: Confound regression ===
code(
    "t0 = time.time()\n"
    "residuals = regress_confounds(bold_2d, confounds_aligned)\n"
    "elapsed = time.time() - t0\n"
    'print(f"Confound regression: {elapsed:.1f} s ({residuals.shape[0]:,} voxels)")'
)

# === Cell 20: Confound QC plot ===
code(
    "# Show effect of confound regression on an example voxel\n"
    "voxel_idx = 150000\n"
    "fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)\n"
    "\n"
    't_vol = np.arange(n_triggers) * egg["tr"]\n'
    'axes[0].plot(t_vol, bold_2d[voxel_idx], linewidth=0.5, color="steelblue")\n'
    'axes[0].set_ylabel("Original BOLD")\n'
    'axes[0].set_title(f"Voxel {voxel_idx:,}: Before vs. After Confound Regression")\n'
    "\n"
    'axes[1].plot(t_vol, residuals[voxel_idx], linewidth=0.5, color="#27AE60")\n'
    'axes[1].set_ylabel("Residuals (z-scored)")\n'
    'axes[1].set_xlabel("Time (s)")\n'
    "\n"
    "for ax in axes:\n"
    '    ax.spines["top"].set_visible(False)\n'
    '    ax.spines["right"].set_visible(False)\n'
    "\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# === Cell 21: BOLD phase section ===
md(
    "### 6b. BOLD Phase Extraction\n"
    "\n"
    "Bandpass filter each voxel at the same gastric frequency as the EGG,\n"
    "then extract instantaneous phase via Hilbert transform.\n"
    "\n"
    "We use an IIR (Butterworth) filter because BOLD time series are too\n"
    "short (~420 volumes) for the FIR filter that works at EGG sampling\n"
    "rates. The vectorized IIR path processes all ~350K voxels at once."
)

# === Cell 22: BOLD phases ===
code(
    "t0 = time.time()\n"
    "bold_phases = bold_voxelwise_phases(\n"
    "    residuals,\n"
    "    peak_freq,\n"
    '    sfreq=1 / egg["tr"],\n'
    "    begin_cut=begin_cut,\n"
    "    end_cut=end_cut,\n"
    ")\n"
    "elapsed = time.time() - t0\n"
    'print(f"BOLD phase extraction: {elapsed:.1f} s ({residuals.shape[0]:,} voxels)")\n'
    'print(f"BOLD phases shape: {bold_phases.shape}")\n'
    'print(f"EGG phase length:  {len(egg_phase)} (should match)")'
)

# === Cell 23: PLV section ===
md(
    "## 7. Compute PLV Map\n"
    "\n"
    "Phase-locking value (PLV) between EGG phase and each BOLD voxel's\n"
    "phase. Values range from 0 (no coupling) to 1 (perfect phase locking).\n"
    "\n"
    "We pass the artifact mask to exclude censored volumes from the PLV\n"
    "computation."
)

# === Cell 24: PLV map ===
code(
    "plv_3d = compute_plv_map(\n"
    "    egg_phase,\n"
    "    bold_phases,\n"
    "    vol_shape=vol_shape,\n"
    "    mask_indices=mask_data,\n"
    "    artifact_mask=vol_mask,\n"
    ")\n"
    "\n"
    "plv_flat = plv_3d[mask_data]\n"
    "\n"
    'print(f"PLV volume shape: {plv_3d.shape}")\n'
    'print(f"Volumes used: {vol_mask.sum()} / {len(vol_mask)} (after censoring)")\n'
    'print("PLV statistics (brain voxels only):")\n'
    'print(f"  Mean:   {plv_flat.mean():.4f}")\n'
    'print(f"  Median: {np.median(plv_flat):.4f}")\n'
    'print(f"  Max:    {plv_flat.max():.4f}")\n'
    'print(f"  Std:    {plv_flat.std():.4f}")'
)

# === Cell 25: PLV stat map ===
code(
    "plv_img = to_nifti(plv_3d, affine)\n"
    "\n"
    "# Stat map overlay on MNI template\n"
    'display = gp.plot_coupling_map(plv_img, threshold=0.04, title="Empirical PLV Map")\n'
    "plt.show()"
)

# === Cell 26: PLV glass brain ===
code(
    "# Glass brain (transparent overview)\n"
    'display = gp.plot_glass_brain(plv_img, threshold=0.04, title="Empirical PLV \\u2014 Glass Brain")\n'
    "plt.show()"
)

# === Cell 27: PLV histogram ===
code(
    "fig, ax = plt.subplots(figsize=(8, 3))\n"
    'ax.hist(plv_flat, bins=50, color="steelblue", edgecolor="white", alpha=0.8)\n'
    'ax.axvline(plv_flat.mean(), color="crimson", linewidth=2, label=f"Mean = {plv_flat.mean():.4f}")\n'
    'ax.set_xlabel("PLV")\n'
    'ax.set_ylabel("Voxel count")\n'
    'ax.set_title("Distribution of Empirical PLV Across Brain Voxels")\n'
    "ax.legend()\n"
    'ax.spines["top"].set_visible(False)\n'
    'ax.spines["right"].set_visible(False)\n'
    "plt.tight_layout()\n"
    "plt.show()"
)

# === Cell 28: Surrogate section ===
md(
    "## 8. Surrogate Statistical Testing\n"
    "\n"
    "Observed PLV may be non-zero by chance due to autocorrelation. We test\n"
    "significance using the **circular time-shift** method: shift the EGG\n"
    "phase by random offsets and recompute PLV to build a null distribution.\n"
    "\n"
    "We use 200 surrogates (publication-quality). The artifact mask is\n"
    "applied consistently to each surrogate shift."
)

# === Cell 29: Surrogate PLV ===
code(
    "t0 = time.time()\n"
    "surr_3d = compute_surrogate_plv_map(\n"
    "    egg_phase,\n"
    "    bold_phases,\n"
    "    vol_shape=vol_shape,\n"
    "    mask_indices=mask_data,\n"
    "    n_surrogates=200,\n"
    "    seed=42,\n"
    "    artifact_mask=vol_mask,\n"
    ")\n"
    "elapsed = time.time() - t0\n"
    "surr_flat = surr_3d[mask_data]\n"
    "\n"
    'print(f"Surrogate PLV: {elapsed:.1f} s (200 circular shifts)")\n'
    'print("Surrogate PLV (brain voxels):")\n'
    'print(f"  Mean: {surr_flat.mean():.4f}")\n'
    'print(f"  Max:  {surr_flat.max():.4f}")'
)

# === Cell 30: Z-score ===
code(
    "# Z-score: empirical - surrogate (positive = true coupling)\n"
    "z_3d = np.zeros_like(plv_3d)\n"
    "z_3d[mask_data] = gp.coupling_zscore(plv_flat, surr_flat)\n"
    "z_flat = z_3d[mask_data]\n"
    "\n"
    'print("Coupling z-score (brain voxels):")\n'
    'print(f"  Mean:  {z_flat.mean():.4f}")\n'
    'print(f"  Max:   {z_flat.max():.4f}")\n'
    'print(f"  >0.01: {(z_flat > 0.01).sum():,} voxels ({(z_flat > 0.01).mean():.1%})")'
)

# === Cell 31: Z-score stat map ===
code(
    "z_img = to_nifti(z_3d, affine)\n"
    "\n"
    "display = gp.plot_coupling_map(\n"
    "    z_img, threshold=0.01,\n"
    '    title="Coupling Z-Score (Empirical \\u2212 Surrogate)",\n'
    '    cmap="YlOrRd",\n'
    ")\n"
    "plt.show()"
)

# === Cell 32: Z-score glass brain ===
code(
    "display = gp.plot_glass_brain(\n"
    "    z_img, threshold=0.01,\n"
    '    title="Coupling Z-Score \\u2014 Glass Brain",\n'
    ")\n"
    "plt.show()"
)

# === Cell 33: Comparison histogram ===
code(
    "fig, ax = plt.subplots(figsize=(8, 3))\n"
    'ax.hist(plv_flat, bins=50, alpha=0.6, color="steelblue", edgecolor="white", label="Empirical PLV")\n'
    'ax.hist(surr_flat, bins=50, alpha=0.6, color="grey", edgecolor="white", label="Surrogate PLV (median)")\n'
    'ax.set_xlabel("PLV")\n'
    'ax.set_ylabel("Voxel count")\n'
    'ax.set_title("Empirical vs. Surrogate PLV Distribution")\n'
    "ax.legend()\n"
    'ax.spines["top"].set_visible(False)\n'
    'ax.spines["right"].set_visible(False)\n'
    "plt.tight_layout()\n"
    "plt.show()"
)

# === Cell 34: Summary ===
md(
    "## 9. Summary\n"
    "\n"
    "This tutorial demonstrated the complete Rebollo et al. gastric-brain\n"
    "coupling pipeline on real fMRIPrep data:\n"
    "\n"
    "| Step | Function | Time |\n"
    "|------|----------|------|\n"
    "| Load + smooth BOLD | nibabel + ``smooth_img(fwhm=6)`` | ~10 s |\n"
    "| Align volumes | ``align_bold_to_egg`` | instant |\n"
    "| EGG channel selection | ``select_best_channel`` | instant |\n"
    "| EGG bandpass + phase | ``apply_bandpass`` + ``instantaneous_phase`` | instant |\n"
    "| Per-volume phase | ``phase_per_volume`` + ``apply_volume_cuts`` | instant |\n"
    "| Artifact detection | ``detect_phase_artifacts`` + ``artifact_mask_to_volumes`` | instant |\n"
    "| Confound regression | ``regress_confounds`` | ~5 s |\n"
    "| BOLD phase extraction | ``bold_voxelwise_phases`` (IIR, vectorized) | ~7 s |\n"
    "| PLV map (with mask) | ``compute_plv_map`` | ~1 s |\n"
    "| Surrogate testing (with mask) | ``compute_surrogate_plv_map`` (200 shifts) | ~10 min |\n"
    "| Visualization | ``plot_coupling_map`` / ``plot_glass_brain`` | instant |\n"
    "\n"
    "**Key parameters:**\n"
    "- EGG peak frequency: individual (data-driven)\n"
    "- Filter bandwidth: peak \u00b1 0.015 Hz (HWHM)\n"
    "- Spatial smoothing: 6 mm FWHM Gaussian\n"
    "- Volume trimming: 21 from each edge\n"
    "- Artifact censoring: volumes with bad EGG phase excluded\n"
    "- Confounds: 6 motion + 6 aCompCor (12 regressors)\n"
    "- BOLD filter: IIR Butterworth order 4 (vectorized)\n"
    "\n"
    "**For publication-quality results:**\n"
    '- Consider using the full surrogate distribution (``stat="all"``)\n'
    "  to compute permutation p-values and FDR correction"
)

# === Assemble notebook ===
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("docs/tutorials/fmri_coupling_real.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created notebook with {len(cells)} cells")
