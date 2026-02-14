API Reference
=============

GastroPy exposes a flat namespace: most functions are accessible directly
from ``import gastropy as gp``. The reference below is organized by
submodule.

.. contents:: Modules
   :local:
   :depth: 1

.. currentmodule:: gastropy

EGG --- Core Pipeline
---------------------

High-level functions for EGG signal processing.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   egg_process
   egg_clean
   select_best_channel
   select_peak_frequency

Signal --- DSP Utilities
------------------------

Low-level signal processing building blocks.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   psd_welch
   design_fir_bandpass
   apply_bandpass
   instantaneous_phase
   cycle_durations
   mean_phase_per_window
   resample_signal
   detect_phase_artifacts
   find_cycle_edges

Metrics --- Rhythm Quantification
---------------------------------

Gastric rhythm metrics and quality assessment.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   band_power
   instability_coefficient
   cycle_stats
   proportion_normogastric
   assess_quality
   GastricBand
   NORMOGASTRIA
   BRADYGASTRIA
   TACHYGASTRIA
   GASTRIC_BANDS

Time-Frequency --- Per-Band Decomposition
------------------------------------------

Decompose signals across gastric frequency bands.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   band_decompose
   multiband_analysis

Coupling --- Phase Coupling Analysis
-------------------------------------

Phase-locking value (PLV), surrogate testing, and circular statistics
for gastric-brain coupling analyses.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   phase_locking_value
   phase_locking_value_complex
   surrogate_plv
   coupling_zscore
   circular_mean
   resultant_length
   rayleigh_test

Data --- Sample Datasets
------------------------

Load bundled example EGG recordings for tutorials and testing.
``fetch_fmri_bold`` downloads preprocessed fMRI data from a GitHub Release.

.. currentmodule:: gastropy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   load_fmri_egg
   load_egg
   list_datasets
   fetch_fmri_bold

Viz --- Visualization
---------------------

Plotting functions for EGG signals and gastric rhythm analysis.

.. currentmodule:: gastropy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   plot_psd
   plot_egg_overview
   plot_cycle_histogram
   plot_artifacts
   plot_volume_phase
   plot_egg_comprehensive

Neuro --- Neuroimaging Utilities
--------------------------------

Functions specific to gastric-brain coupling with fMRI, EEG, and MEG.
Access via ``gastropy.neuro.fmri``.

.. currentmodule:: gastropy.neuro.fmri

.. autosummary::
   :toctree: generated/
   :nosignatures:

   find_scanner_triggers
   create_volume_windows
   phase_per_volume
   apply_volume_cuts
   regress_confounds
   bold_voxelwise_phases
   compute_plv_map
   compute_surrogate_plv_map
