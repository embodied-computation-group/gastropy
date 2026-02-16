---
title: "GastroPy: A Python Package for Electrogastrography Signal Processing and Gastric-Brain Coupling Analysis"
tags:
  - Python
  - electrogastrography
  - EGG
  - gastric rhythm
  - brain-gut coupling
  - signal processing
  - neuroscience
  - fMRI
authors:
  - name: Micah Allen
    orcid: 0000-0001-9399-4179
    corresponding: true
    affiliation: 1
  - name: Daniel Kluger
    affiliation: "TODO"
  - name: Leah Banellis
    affiliation: "TODO"
  - name: Ignacio Rebollo
    affiliation: "TODO"
  - name: Nils Kroemer
    affiliation: "TODO"
  - name: Edwin Dalmaijer
    affiliation: "TODO"
affiliations:
  - name: Aarhus University, Institute of Clinical Medicine, Denmark
    index: 1
date: 16 February 2026
bibliography: paper.bib
---

# Summary

GastroPy is an open-source Python package for processing electrogastrography
(EGG) recordings and computing gastric-brain coupling metrics. EGG is a
non-invasive technique that measures gastric myoelectrical activity via
cutaneous electrodes placed on the abdomen, capturing the ~0.05 Hz (3 cycles
per minute) slow wave that governs gastric motility [@Koch2004]. GastroPy
provides a modular pipeline spanning spectral analysis, bandpass filtering,
instantaneous phase extraction, cycle-level metrics, multi-channel selection,
artifact detection, time-frequency decomposition, and phase-locking value
(PLV) computation for gastric-brain coupling with fMRI.

The package is built on NumPy and SciPy for core digital signal processing,
with optional dependencies on MNE-Python [@Gramfort2013], nilearn
[@Abraham2014], and nibabel for neuroimaging workflows. GastroPy is designed
for researchers studying the brain-gut axis who need reproducible,
well-tested tools that integrate with the broader Python scientific
ecosystem.

# Statement of Need

<!-- TODO: Expand to ~300-400 words -->

Electrogastrography has seen renewed interest as a tool for studying
brain-body interactions, interoception, and the gut-brain axis
[@Wolpert2020; @Rebollo2018]. Despite this growth, the field lacks a
dedicated, open-source Python package for EGG analysis. Researchers
typically rely on custom MATLAB scripts [@Banellis2025], ad hoc Python
code, or general-purpose biosignal toolkits that offer limited EGG-specific
functionality.

Existing Python packages address adjacent needs but leave significant gaps.
NeuroKit2 [@Makowski2021] provides general biosignal processing with basic
EGG support, but lacks multi-channel selection, gastric-brain coupling
pipelines, and EGG-specific artifact detection. MNE-Python [@Gramfort2013]
excels at EEG and MEG analysis but does not target the low-frequency
(0.01--0.1 Hz) gastric rhythm. MATLAB toolboxes such as the StomachBrain
pipeline [@Banellis2025] are not accessible to the growing number of
researchers working in Python, and their monolithic design limits
reusability.

GastroPy fills this gap by providing:

- A complete EGG processing pipeline from raw signal to publication-ready
  metrics, with sensible defaults and full parameter control.
- Multi-channel selection using spectral peak detection in the normogastric
  band (2--4 cycles per minute).
- Phase-based artifact detection adapted from @Wolpert2020.
- Modality-agnostic phase-locking value computation for gastric-brain
  coupling, with surrogate-based statistical testing.
- A dedicated fMRI sub-module for scanner trigger alignment, confound
  regression, voxelwise BOLD phase extraction, and whole-brain PLV map
  generation.

GastroPy is intended for neuroscientists, gastroenterologists, and
psychophysiology researchers who work with EGG data, whether standalone
or concurrent with fMRI, EEG, or MEG.

# State of the Field

<!-- TODO: Expand comparison table and build-vs-contribute justification -->

| Package | Language | EGG Pipeline | Multi-channel | Coupling | Artifact Detection |
|---------|----------|-------------|---------------|----------|--------------------|
| GastroPy | Python | Full | Yes | PLV + surrogates | Phase-based |
| NeuroKit2 | Python | Basic | No | No | General |
| MNE-Python | Python | No | N/A | No | EEG/MEG-focused |
| StomachBrain | MATLAB | Partial | No | PLV | No |

We chose to build a new package rather than contribute to an existing one
because the EGG processing pipeline requires domain-specific decisions
(frequency bands, cycle detection, phase artifact criteria) that are
difficult to retrofit into general-purpose toolkits. GastroPy's layered
architecture isolates core DSP from neuroimaging-specific logic, making it
usable both as a standalone EGG toolkit and as a component in multimodal
brain-body pipelines.

# Software Design

<!-- TODO: Expand to ~200-300 words -->

GastroPy follows a layered, modular architecture inspired by NeuroKit2's
flat API design. The package is organized into seven core modules:

- **`gastropy.signal`**: Low-level DSP functions (PSD, filtering, phase
  extraction, resampling) operating on plain NumPy arrays with no
  external dependencies beyond SciPy.
- **`gastropy.metrics`**: Gastric rhythm quantification (band power,
  instability coefficient, cycle statistics, quality assessment).
- **`gastropy.egg`**: High-level pipeline (`egg_process`) composing
  signal and metrics functions into a single-call workflow.
- **`gastropy.timefreq`**: Time-frequency decomposition via narrowband
  filtering and Morlet wavelets.
- **`gastropy.coupling`**: Modality-agnostic circular statistics and PLV
  computation, usable with any paired phase time series.
- **`gastropy.neuro.fmri`**: fMRI-specific convenience functions
  (trigger detection, volume windowing, confound regression, BOLD
  phase extraction, PLV map generation).
- **`gastropy.viz`**: Publication-ready plotting functions for all
  analysis stages.

A key design decision is dependency isolation: core signal processing
requires only NumPy, SciPy, pandas, and matplotlib, while neuroimaging
features (MNE, nilearn, nibabel) are optional extras. This keeps the
base install lightweight for users who only need EGG processing.

All public functions accept and return NumPy arrays or pandas DataFrames,
ensuring composability with the broader scientific Python ecosystem. The
API is designed so that each function can be used independently or
composed into custom pipelines beyond what `egg_process` provides.

# Research Impact

<!-- TODO: Expand with concrete citations and adoption evidence -->

GastroPy's core algorithms were developed and validated in the context of
published research on gastric-brain coupling and interoception. The signal
processing pipeline was ported from analysis code used in studies of
precision and brain-gut interaction, and the coupling module implements
the PLV methodology described in @Banellis2025.

The package is designed to support reproducible research by providing
tested, documented implementations of methods that are currently scattered
across lab-specific scripts. By lowering the barrier to rigorous EGG
analysis, GastroPy aims to accelerate research on the brain-gut axis
and facilitate cross-lab replication.

# AI Usage Disclosure

Generative AI tools were used during the development of GastroPy:

- **Tool**: Anthropic Claude (Claude Code CLI), models claude-sonnet-4-20250514
  and claude-opus-4-20250514.
- **Scope**: AI assistance was used for code generation, test writing,
  documentation drafting, and debugging across all modules.
- **Human oversight**: All AI-generated code was reviewed, tested, and
  validated by the authors. Architectural decisions, algorithm selection,
  and scientific methodology were determined by the authors. The automated
  test suite (216 tests) verifies correctness of all implementations.

The authors accept full responsibility for the accuracy, originality, and
licensing of all code and documentation in this package.

# Acknowledgements

<!-- TODO: Add funding sources, grants, institutional support -->

We acknowledge support from [funding agency / grant number].

# References
