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
| **`gastropy.signal`** | Welch PSD, FIR/IIR bandpass filtering, Hilbert phase extraction, cycle detection, resampling |
| **`gastropy.metrics`** | Gastric frequency bands, band power, instability coefficient, cycle statistics, quality assessment |
| **`gastropy.egg`** | High-level pipeline (`egg_process`), channel selection, peak frequency detection |
| **`gastropy.neuro.fmri`** | Scanner trigger parsing, volume windowing, per-volume phase, transient removal |

> **Planned:** time-frequency analysis, gastric-brain phase coupling,
> visualization, data I/O with BIDS support, and sample datasets.
> See the [Roadmap](#roadmap) below.

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

print(signals.head())
#        raw  filtered     phase  amplitude
# 0  0.0175    0.0012 -2.356194     0.0038
# ...

print(f"Peak frequency : {info['peak_freq_hz']:.3f} Hz")
print(f"Cycles detected: {info['cycle_stats']['n_cycles']}")
print(f"Instability IC : {info['instability_coefficient']:.4f}")
print(f"% Normogastric : {info['proportion_normogastric']:.0%}")
```

### Step-by-step control

```python
# Spectral analysis
freqs, psd = gp.psd_welch(signal, sfreq, fmin=0.01, fmax=0.1)

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
```

## Documentation

Full API reference and tutorials at
**[embodied-computation-group.github.io/gastropy](https://embodied-computation-group.github.io/gastropy)**.

## Roadmap

GastroPy is under active development. Current status:

- [x] `gastropy.signal` — core DSP (PSD, filtering, phase, resampling)
- [x] `gastropy.metrics` — band power, instability coefficient, quality control
- [x] `gastropy.egg` — high-level pipeline, channel selection
- [x] `gastropy.neuro.fmri` — scanner triggers, volume windowing
- [ ] `gastropy.timefreq` — wavelets, spectrograms
- [ ] `gastropy.coupling` — gastric-brain phase coupling
- [ ] `gastropy.viz` — publication-ready plotting
- [ ] `gastropy.io` — data I/O, BIDS support
- [ ] `gastropy.neuro.eeg` / `gastropy.neuro.meg` — EEG/MEG utilities
- [ ] `gastropy.stats` — statistical testing
- [ ] `gastropy.data` — sample datasets

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
