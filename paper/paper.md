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
    affiliation: "1, 2"
  - name: Daniel Kluger
    affiliation: 3
  - name: Leah Banellis
    affiliation: "1, 2"
  - name: Ignacio Rebollo
    affiliation: 4
  - name: Nils Kroemer
    affiliation: 5
  - name: Edwin Dalmaijer
    affiliation: 6
affiliations:
  - name: Center of Functionally Integrative Neuroscience, Aarhus University, Denmark
    index: 1
  - name: Cambridge Psychiatry, University of Cambridge, United Kingdom
    index: 2
  - name: Institute of Biomagnetism and Biosignal Analysis, University of Muenster, Germany
    index: 3
  - name: German Institute for Human Nutrition, Potsdam, Germany
    index: 4
  - name: Department of Psychiatry and Psychotherapy, University of Tuebingen, Germany
    index: 5
  - name: School of Psychology, University of Leeds, United Kingdom
    index: 6
date: 16 February 2026
bibliography: paper.bib
---

# Summary

GastroPy is an open-source Python package for processing electrogastrography
(EGG) recordings and computing gastric-brain coupling metrics. EGG is a
non-invasive technique that measures gastric myoelectrical activity via
cutaneous electrodes placed on the abdomen, capturing the ~0.05 Hz (3 cycles
per minute) slow wave that governs gastric motility [@Koch2004;
@ChenMcCallum1991]. GastroPy provides a modular pipeline spanning spectral
analysis, bandpass filtering, instantaneous phase extraction, cycle-level
metrics, multi-channel selection, artifact detection, time-frequency
decomposition, and phase-locking value (PLV) computation for gastric-brain
coupling with fMRI.

The package is built on NumPy [@Harris2020] and SciPy [@Virtanen2020] for core
digital signal processing, with optional dependencies on MNE-Python
[@Gramfort2013], nilearn [@Abraham2014], and nibabel for neuroimaging
workflows. GastroPy is designed for researchers studying the brain-gut axis who
need reproducible, well-tested tools that integrate with the broader Python
scientific ecosystem.

# Statement of Need

Electrogastrography has seen renewed interest as a tool for studying
brain-body interactions, interoception, and the gut-brain axis. Landmark
studies have demonstrated phase synchrony between the gastric rhythm and
resting-state brain networks [@Rebollo2018], established practical recording
and analysis guidelines for psychophysiology research [@Wolpert2020], and
linked gastric-brain coupling to dimensional signatures of mental health
[@Banellis2025]. Concurrent EGG-fMRI studies have revealed that multiple
resting-state networks exhibit phase-locking with the gastric basal electrical
rhythm [@Choe2021], while methodological work has begun to characterize the
reliability and validity of these coupling measures [@Levakov2023]. Beyond
neuroscience, wearable EGG devices are being explored for affect detection
and emotion regulation applications [@Vujic2020].

Despite this growth, the field lacks a dedicated, open-source Python package
for EGG analysis. Researchers typically rely on custom MATLAB scripts
[@Banellis2025], ad hoc Python code, or general-purpose biosignal toolkits
that offer limited EGG-specific functionality. This fragmentation impedes
reproducibility, slows adoption by new labs, and makes it difficult to compare
results across studies that use different analysis pipelines.

Existing Python packages address adjacent needs but leave significant gaps.
NeuroKit2 [@Makowski2021] provides general biosignal processing with basic EGG
support, but lacks multi-channel selection, gastric-brain coupling pipelines,
and EGG-specific artifact detection. MNE-Python [@Gramfort2013] excels at EEG
and MEG analysis but does not target the low-frequency (0.01--0.1 Hz) gastric
rhythm. Related packages such as Systole [@Legrand2020] and HeartPy
[@vanGent2019] demonstrate the value of domain-specific biosignal toolkits but
focus exclusively on cardiac signals. MATLAB toolboxes such as the
StomachBrain pipeline [@Banellis2025] are not accessible to the growing number
of researchers working in Python, and their monolithic design limits
reusability.

GastroPy fills this gap by providing:

- A complete EGG processing pipeline from raw signal to publication-ready
  metrics, with sensible defaults and full parameter control.
- Multi-channel selection using spectral peak detection in the normogastric
  band (2--4 cycles per minute), following established clinical guidelines
  [@Chang2005; @Yin2013].
- Phase-based artifact detection adapted from @Wolpert2020, identifying
  non-monotonic phase progressions and duration outliers at the cycle level.
- Modality-agnostic phase-locking value computation for gastric-brain
  coupling, with surrogate-based statistical testing via circular
  time-shifting.
- A dedicated fMRI sub-module for scanner trigger alignment, confound
  regression, voxelwise BOLD phase extraction, and whole-brain PLV map
  generation, implementing the methodology described in @Banellis2025.

GastroPy is intended for neuroscientists, gastroenterologists, and
psychophysiology researchers who work with EGG data, whether standalone or
concurrent with fMRI, EEG, or MEG.

# State of the Field

\autoref{tab:comparison} summarizes the capabilities of existing tools
relative to GastroPy.

| Package | Language | EGG Pipeline | Multi-channel | Coupling | Artifact Detection |
|---------|----------|-------------|---------------|----------|--------------------|
| GastroPy | Python | Full | Yes | PLV + surrogates | Phase-based |
| NeuroKit2 | Python | Basic | No | No | General |
| MNE-Python | Python | No | N/A | No | EEG/MEG-focused |
| StomachBrain | MATLAB | Partial | No | PLV | No |

: Comparison of EGG analysis tools. {#tab:comparison}

We chose to build a new package rather than contribute to an existing one
because the EGG processing pipeline requires domain-specific decisions
(frequency bands, cycle detection, phase artifact criteria) that are difficult
to retrofit into general-purpose toolkits. The gastric slow wave occupies a
unique frequency range (0.03--0.07 Hz) that falls below the passband of most
electrophysiology tools, and its coupling with brain signals requires
specialized windowing and surrogate testing procedures. GastroPy's layered
architecture isolates core DSP from neuroimaging-specific logic, making it
usable both as a standalone EGG toolkit and as a component in multimodal
brain-body pipelines.

# Software Design

GastroPy follows a layered, modular architecture inspired by NeuroKit2's flat
API design [@Makowski2021]. The package is organized into seven core modules:

- **`gastropy.signal`**: Low-level DSP functions (PSD via Welch's method,
  FIR/IIR bandpass filtering, Hilbert-based phase extraction, resampling)
  operating on plain NumPy arrays with no external dependencies beyond SciPy.
- **`gastropy.metrics`**: Gastric rhythm quantification including band power,
  instability coefficient [@Koch2004], cycle statistics, proportion
  normogastric, and automated quality assessment.
- **`gastropy.egg`**: High-level pipeline functions (`egg_process`,
  `egg_clean`) composing signal and metrics functions into single-call
  workflows, plus `select_best_channel` for multi-channel recordings.
- **`gastropy.timefreq`**: Time-frequency decomposition via narrowband
  filtering and Morlet wavelets, enabling analysis of gastric rhythm dynamics
  across bradygastric, normogastric, and tachygastric bands.
- **`gastropy.coupling`**: Modality-agnostic circular statistics (circular
  mean, resultant length, Rayleigh test) and PLV computation with
  surrogate-based z-scoring, usable with any paired phase time series.
- **`gastropy.neuro.fmri`**: fMRI-specific convenience functions for trigger
  detection, volume windowing, confound regression, BOLD phase extraction, and
  whole-brain PLV map generation with surrogate distributions.
- **`gastropy.viz`**: Publication-ready plotting functions built on matplotlib
  [@Hunter2007] for all analysis stages, from PSD plots to comprehensive
  multi-panel figures.

A key design decision is dependency isolation: core signal processing requires
only NumPy, SciPy, pandas [@McKinney2010], and matplotlib, while neuroimaging
features (MNE, nilearn, nibabel) are optional extras installed via
`pip install gastropy[neuro]`. This keeps the base install lightweight for
users who only need EGG processing.

All public functions accept and return NumPy arrays or pandas DataFrames,
ensuring composability with the broader scientific Python ecosystem. The API is
designed so that each function can be used independently or composed into
custom pipelines beyond what `egg_process` provides. The package includes
bundled sample data (three fMRI-EGG sessions and one standalone EGG recording)
and supports downloading larger BOLD datasets via the `fetch_fmri_bold`
function.

# Research Impact

GastroPy's core algorithms were developed and validated in the context of
published research on gastric-brain coupling and interoception. The signal
processing pipeline was ported from analysis code used in studies of precision
and brain-gut interaction, and the coupling module implements the PLV
methodology described in @Banellis2025, which demonstrated that frontoparietal
brain coupling to the gastric rhythm indexes a dimensional signature of mental
health across 243 participants. The artifact detection algorithms follow the
phase-based criteria established by @Wolpert2020 in a large normative sample
(N=117), while the spectral analysis parameters align with clinical EGG
guidelines [@Chang2005; @Yin2013; @ChenMcCallum1991].

The package is designed to support reproducible research by providing tested,
documented implementations of methods that are currently scattered across
lab-specific scripts. By lowering the barrier to rigorous EGG analysis,
GastroPy aims to accelerate research on the brain-gut axis and facilitate
cross-lab replication. The automated test suite (178 tests) verifies
correctness of all signal processing, metric computation, coupling analysis,
and visualization functions.

# AI Usage Disclosure

Generative AI tools were used during the development of GastroPy:

- **Tool**: Anthropic Claude (Claude Code CLI), models claude-sonnet-4-20250514
  and claude-opus-4-20250514.
- **Scope**: AI assistance was used for code generation, test writing,
  documentation drafting, and debugging across all modules.
- **Human oversight**: All AI-generated code was reviewed, tested, and
  validated by the authors. Architectural decisions, algorithm selection, and
  scientific methodology were determined by the authors. The automated test
  suite verifies correctness of all implementations.

The authors accept full responsibility for the accuracy, originality, and
licensing of all code and documentation in this package.

# Acknowledgements

We acknowledge support from the Lundbeck Foundation (R380-2021-1538, R140-2013-13057),
the European Research Council (ERC-2020-COG, 101001893, EMBODIED-COMPUTATION),
and Aarhus University.

# References
