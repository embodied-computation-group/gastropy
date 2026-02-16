# Changelog

All notable changes to GastroPy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`gastropy.io`** — BIDS peripheral physiology I/O module:
  `read_bids_physio`, `write_bids_physio`, `parse_bids_filename` (stdlib +
  numpy only), and `brainvision_to_bids` converter (optional MNE dependency).
- 19 new tests for the IO module (read, write, round-trip, error handling,
  BIDS filename parsing, BrainVision import guard).

### Changed

- **Sample data migrated from NPZ to BIDS physio format.** All bundled
  datasets are now stored as `_physio.tsv.gz` + `_physio.json` sidecar pairs
  following the BIDS peripheral physiology specification. The public API
  (`load_egg`, `load_fmri_egg`, `list_datasets`) returns identical results.
- `gastropy.data` internals now use `read_bids_physio` instead of `np.load`.
- `pyproject.toml` build artifacts glob updated for BIDS file extensions.

### Removed

- Legacy NPZ sample data files (`egg_standalone.npz`,
  `fmri_egg_session_*.npz`).

## [0.1.0] - 2026-02-16

Initial public release of GastroPy.

### Added

- **`gastropy.signal`** — Welch PSD (configurable overlap), FIR bandpass
  filter design and application, Hilbert-based instantaneous phase and
  amplitude, cycle duration extraction, mean phase per window, signal
  resampling, phase-based artifact detection, and cycle edge finding.
- **`gastropy.metrics`** — Gastric frequency band definitions
  (`GastricBand`, `NORMOGASTRIA`, `BRADYGASTRIA`, `TACHYGASTRIA`), band
  power computation, instability coefficient, cycle statistics,
  proportion normogastric, and signal quality assessment.
- **`gastropy.egg`** — High-level `egg_process` pipeline, `egg_clean`
  preprocessing, multi-channel selection via `select_best_channel`, and
  peak frequency detection via `select_peak_frequency`.
- **`gastropy.timefreq`** — Per-band narrowband decomposition
  (`band_decompose`), multi-band analysis across brady/normo/tachy bands,
  and Morlet wavelet time-frequency representation (`morlet_tfr`).
- **`gastropy.coupling`** — Phase-locking value (PLV and complex PLV),
  surrogate PLV generation, coupling z-score, circular mean, resultant
  vector length, and Rayleigh test for non-uniformity.
- **`gastropy.neuro.fmri`** — Scanner trigger detection, volume window
  creation, phase-per-volume extraction, volume cut application, confound
  regression, voxelwise BOLD phase extraction, PLV and surrogate PLV map
  computation, and NIfTI I/O helpers.
- **`gastropy.data`** — Bundled sample datasets: 3 fMRI-EGG sessions
  (8-channel, 10 Hz) and 1 standalone EGG recording (7-channel, 10 Hz,
  Wolpert et al. 2020). Downloadable fMRI BOLD data via `fetch_fmri_bold`
  (pooch-based, ~1.2 GB from GitHub Releases).
- **`gastropy.viz`** — Publication-ready plotting: `plot_psd`,
  `plot_egg_overview`, `plot_cycle_histogram`, `plot_artifacts`,
  `plot_volume_phase`, `plot_egg_comprehensive`, `plot_tfr`,
  `plot_coupling_map`, and `plot_glass_brain`.
- **Documentation** — Sphinx site with sphinx-book-theme, full API
  reference (58 auto-generated pages), 3 narrative tutorials, and 20
  focused example notebooks.
- **Testing** — 216 tests across 11 test modules, CI on Python
  3.10/3.11/3.12/3.13, Codecov integration.
- **Infrastructure** — Hatch build system, Ruff linting and formatting,
  pre-commit hooks, GitHub Actions for tests/lint/docs.

[Unreleased]: https://github.com/embodied-computation-group/gastropy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/embodied-computation-group/gastropy/releases/tag/v0.1.0
