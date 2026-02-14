<p align="center">
  <img src="gastropy_logo.png" alt="GastroPy Logo" width="400">
</p>

<h1 align="center">GastroPy</h1>

<p align="center">
  <em>A Python toolkit for electrogastrography (EGG) signal processing and gastric-brain coupling analysis.</em>
</p>

<p align="center">
  <a href="https://github.com/embodied-computation-group/gastropy/actions/workflows/tests.yml"><img src="https://github.com/embodied-computation-group/gastropy/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://embodied-computation-group.github.io/gastropy"><img src="https://github.com/embodied-computation-group/gastropy/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://github.com/embodied-computation-group/gastropy/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
</p>

---

## What is GastroPy?

GastroPy gives researchers a modular, tested Python pipeline for working
with electrogastrography data — from raw signal to publication-ready metrics.
It is designed to work with EGG recordings from **any** acquisition setup
(standalone Biopac, ambulatory, MRI-concurrent, etc.) and provides
specialised helpers for gastric-brain coupling analyses with fMRI, EEG,
and MEG.

### Key Features

| Module | What it does |
|---|---|
| **`gastropy.signal`** | Welch PSD, FIR/IIR bandpass filtering, Hilbert phase extraction, cycle detection, resampling, phase-based artifact detection |
| **`gastropy.metrics`** | Gastric frequency bands, band power, instability coefficient, cycle statistics, quality assessment |
| **`gastropy.egg`** | High-level pipeline (`egg_process`), channel selection (`find_peaks`), peak frequency detection |
| **`gastropy.timefreq`** | Per-band narrowband decomposition, cycle analysis across brady/normo/tachy bands |
| **`gastropy.coupling`** | Phase-locking value (PLV), complex PLV, surrogate testing, circular statistics (Rayleigh test, resultant length) |
| **`gastropy.viz`** | PSD plots, EGG overview panels, cycle histograms, artifact displays, fMRI volume phase, brain coupling maps |
| **`gastropy.data`** | Bundled sample datasets (fMRI-EGG and standalone EGG), downloadable fMRI BOLD via `fetch_fmri_bold` |
| **`gastropy.neuro.fmri`** | Scanner triggers, volume windowing, confound regression, voxelwise BOLD phase extraction, PLV map computation, NIfTI I/O |

> **Planned:** data I/O with BIDS support, EEG/MEG utilities, and statistical
> testing. See the [Roadmap](#roadmap) below.

## Installation

### From source (recommended during pre-release)

```bash
git clone https://github.com/embodied-computation-group/gastropy.git
cd gastropy
pip install -e ".[dev]"
```

### Optional extras

```bash
pip install -e ".[neuro]"   # adds MNE, nilearn, nibabel
pip install -e ".[docs]"    # adds Sphinx, sphinx-book-theme, etc.
pip install -e ".[all]"     # everything
```

## Quick Start

### Load sample data

```python
import gastropy as gp

# Bundled sample datasets — no external downloads needed
data = gp.load_egg()                        # standalone 7-channel EGG
fmri = gp.load_fmri_egg(session="0001")     # fMRI-concurrent 8-channel EGG
gp.list_datasets()                           # see all available datasets
```

### One-liner pipeline

```python
import numpy as np
import gastropy as gp

# Simulate a 5-minute EGG recording (3 cpm, 10 Hz sampling)
sfreq = 10.0
t = np.arange(0, 300, 1 / sfreq)
signal = np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.random.randn(len(t))

# Full processing in one call
signals, info = gp.egg_process(signal, sfreq)

print(f"Peak frequency : {info['peak_freq_hz']:.3f} Hz")
print(f"Cycles detected: {info['cycle_stats']['n_cycles']}")
print(f"Instability IC : {info['instability_coefficient']:.4f}")
print(f"% Normogastric : {info['proportion_normogastric']:.0%}")
```

### Artifact detection

```python
# Detect phase artifacts (non-monotonic phase + duration outliers)
artifacts = gp.detect_phase_artifacts(signals["phase"].values, t)
print(f"Artifact cycles: {artifacts['n_artifacts']}")
```

### Visualization

```python
# PSD with normogastric band shading
fig, ax = gp.plot_psd(freqs, psd)

# 4-panel EGG overview (raw, filtered, phase, amplitude)
fig, axes = gp.plot_egg_overview(signals, sfreq)

# Cycle duration histogram
fig, ax = gp.plot_cycle_histogram(info["cycle_durations_s"])

# Phase with artifact overlay
fig, ax = gp.plot_artifacts(signals["phase"].values, t, artifacts)
```

### Step-by-step control

```python
# Spectral analysis (overlap parameter for smoothing control)
freqs, psd = gp.psd_welch(signal, sfreq, fmin=0.01, fmax=0.1, overlap=0.75)

# Bandpass filter to normogastric band (2-4 cpm)
filtered, filt_info = gp.apply_bandpass(signal, sfreq, low_hz=0.033, high_hz=0.067)

# Instantaneous phase & amplitude via Hilbert transform
phase, analytic = gp.instantaneous_phase(filtered)

# Cycle detection and metrics
durations = gp.cycle_durations(phase, t)
ic = gp.instability_coefficient(durations)
```

### Multi-channel selection

```python
# data: (n_channels, n_samples) numpy array
best_ch, peak_freq, freqs, psd = gp.select_best_channel(data, sfreq)
```

### fMRI-concurrent EGG

```python
from gastropy.neuro.fmri import (
    find_scanner_triggers, create_volume_windows,
    phase_per_volume, apply_volume_cuts,
)

onsets = find_scanner_triggers(raw.annotations, label="R128")
windows = create_volume_windows(onsets, tr=1.856, n_volumes=420)
phases = phase_per_volume(analytic, windows)
phases = apply_volume_cuts(phases, begin_cut=21, end_cut=21)

# Visualize per-volume phase with cut regions
fig, ax = gp.plot_volume_phase(phases, tr=1.856, cut_start=21, cut_end=21)
```

### Gastric-brain coupling

```python
import gastropy as gp
from gastropy.neuro.fmri import (
    load_bold, align_bold_to_egg, regress_confounds,
    bold_voxelwise_phases, compute_plv_map, to_nifti,
)

# Load preprocessed fMRI data (requires pip install gastropy[neuro])
bold = load_bold("bold_preproc.nii.gz", "brain_mask.nii.gz")
egg = gp.load_fmri_egg(session="0001")

# Align BOLD volumes to EGG triggers, regress confounds
bold_2d, confounds = align_bold_to_egg(bold["bold_2d"], len(egg["trigger_times"]), confounds_df)
residuals = regress_confounds(bold_2d, confounds)

# Extract BOLD phase at individual gastric frequency
bold_phases = bold_voxelwise_phases(residuals, peak_freq_hz=0.05, sfreq=1/1.856)

# Compute voxelwise PLV map
plv_map = compute_plv_map(egg_phase, bold_phases, vol_shape=bold["vol_shape"], mask_indices=bold["mask"])

# Visualize on brain
plv_img = to_nifti(plv_map, bold["affine"])
gp.plot_coupling_map(plv_img, threshold=0.03)
gp.plot_glass_brain(plv_img, threshold=0.03)
```

## Tutorials & Examples

Step-by-step tutorials covering the full pipeline:

- **[EGG Signal Processing](https://embodied-computation-group.github.io/gastropy/tutorials/egg_processing.html)** —
  From raw data to publication-ready metrics using Wolpert et al. (2020) data.
- **[Gastric-Brain Coupling (Concepts)](https://embodied-computation-group.github.io/gastropy/tutorials/gastric_brain_coupling.html)** —
  PLV pipeline overview with synthetic BOLD data.
- **[Real fMRI Coupling Pipeline](https://embodied-computation-group.github.io/gastropy/tutorials/fmri_coupling_real.html)** —
  End-to-end volumetric PLV map from fMRIPrep data with brain visualization.

Browse the **[Examples Gallery](https://embodied-computation-group.github.io/gastropy/examples/index.html)** for
short, focused examples: PSD plots, signal overviews, cycle histograms,
artifact detection, channel selection, multi-band analysis, PLV computation,
surrogate testing, circular statistics, and brain map visualization.

## Documentation

Full API reference and tutorials at
**[embodied-computation-group.github.io/gastropy](https://embodied-computation-group.github.io/gastropy)**.

## Roadmap

GastroPy is under active development. Current status:

- [x] `gastropy.signal` — core DSP, phase-based artifact detection
- [x] `gastropy.metrics` — band power, instability coefficient, quality control
- [x] `gastropy.egg` — high-level pipeline, channel selection
- [x] `gastropy.timefreq` — per-band decomposition and cycle analysis
- [x] `gastropy.coupling` — PLV, complex PLV, surrogate testing, circular statistics
- [x] `gastropy.neuro.fmri` — triggers, confound regression, BOLD phase extraction, PLV maps, NIfTI I/O
- [x] `gastropy.data` — bundled sample datasets + downloadable fMRI BOLD
- [x] `gastropy.viz` — publication-ready plotting (8 functions including brain maps)
- [ ] `gastropy.io` — data I/O, BIDS support
- [ ] `gastropy.neuro.eeg` / `gastropy.neuro.meg` — EEG/MEG utilities
- [ ] `gastropy.stats` — statistical testing

## Contributing

Contributions are welcome! To get started:

```bash
git clone https://github.com/embodied-computation-group/gastropy.git
cd gastropy
pip install -e ".[dev]"
pre-commit install
```

Run the test suite:

```bash
pytest
```

Check code style:

```bash
ruff check gastropy/
ruff format --check gastropy/
```

## Citation

If you use GastroPy in your research, please cite:

```bibtex
@software{gastropy,
  title   = {GastroPy: A Python Package for Electrogastrography Signal Processing and Gastric-Brain Coupling Analysis},
  author  = {Allen, Micah},
  year    = {2026},
  url     = {https://github.com/embodied-computation-group/gastropy}
}
```

## License

GastroPy is released under the [MIT License](LICENSE).
