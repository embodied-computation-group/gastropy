<p align="center">
  <img src="gastropy_logo.png" alt="GastroPy Logo" width="400">
</p>

<h1 align="center">GastroPy</h1>

<p align="center">
  A Python package for electrogastrography (EGG) signal processing and gastric-brain coupling analysis.
</p>

<p align="center">
  <a href="https://github.com/embodied-computation-group/gastropy/actions/workflows/tests.yml"><img src="https://github.com/embodied-computation-group/gastropy/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/embodied-computation-group/gastropy/actions/workflows/docs.yml"><img src="https://github.com/embodied-computation-group/gastropy/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://github.com/embodied-computation-group/gastropy/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
</p>

---

## Overview

**GastroPy** implements a complete scientific pipeline for electrogastrography, including:

- **Data Import** — Read EGG data from common formats with BIDS support
- **Signal Cleaning** — Artifact removal, filtering, and quality control
- **Signal Processing** — Bandpass filtering, resampling, and general signal utilities
- **Signal Extraction** — Extract gastric rhythm components from raw signals
- **Time-Frequency Analysis** — Wavelets, spectrograms, and power spectral density
- **Metric Extraction** — Peak frequency, instability coefficient, cycle duration, and more
- **Gastric-Brain Coupling** — Phase coupling analyses for fMRI, EEG, and MEG data

## Installation

```bash
pip install gastropy
```

For neuroimaging support (fMRI, EEG, MEG):

```bash
pip install gastropy[neuro]
```

For development:

```bash
pip install gastropy[dev]
```

## Quick Example

```python
import numpy as np
import gastropy as gp

# Simulate a 5-minute EGG recording (3 cpm at 10 Hz)
sfreq = 10.0
t = np.arange(0, 300, 1 / sfreq)
signal = np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.random.randn(len(t))

# Run the full processing pipeline
signals, info = gp.egg_process(signal, sfreq)

print(f"Peak frequency:  {info['peak_freq_hz']:.3f} Hz")
print(f"Cycles detected: {info['cycle_stats']['n_cycles']}")
print(f"Instability IC:  {info['instability_coefficient']:.4f}")
```

## Documentation

Full documentation is available at [embodied-computation-group.github.io/gastropy](https://embodied-computation-group.github.io/gastropy).

## Citation

If you use GastroPy in your research, please cite:

```bibtex
@software{gastropy,
  title = {GastroPy: A Python Package for Electrogastrography Signal Processing and Gastric-Brain Coupling Analysis},
  author = {Allen, Micah},
  year = {2026},
  url = {https://github.com/embodied-computation-group/gastropy}
}
```

## License

GastroPy is licensed under the [MIT License](LICENSE).
