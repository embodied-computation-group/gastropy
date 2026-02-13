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
