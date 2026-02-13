Getting Started
===============

GastroPy provides a flat, NeuroKit2-style API: import the package and
call functions directly.

.. code-block:: python

   import gastropy as gp

Sample Datasets
---------------

GastroPy ships with bundled sample recordings so you can start
experimenting immediately — no external downloads needed.

.. code-block:: python

   import gastropy as gp

   # List available datasets
   gp.list_datasets()

   # Standalone 7-channel EGG (Wolpert et al., 2020)
   egg = gp.load_egg()
   print(egg["signal"].shape, egg["sfreq"])

   # fMRI-concurrent 8-channel EGG (3 sessions available)
   fmri = gp.load_fmri_egg(session="0001")
   print(fmri["signal"].shape, fmri["trigger_times"].shape)

Processing an EGG Signal
------------------------

The fastest way to go from a raw EGG signal to a full set of metrics is
``egg_process``. It filters the signal, extracts instantaneous phase and
amplitude, detects gastric cycles, and computes standard metrics.

.. code-block:: python

   import numpy as np
   import gastropy as gp

   # --- Simulate a 5-minute EGG recording at 10 Hz ---
   sfreq = 10.0
   t = np.arange(0, 300, 1 / sfreq)
   freq_hz = 0.05  # 3 cycles per minute
   signal = np.sin(2 * np.pi * freq_hz * t) + 0.1 * np.random.randn(len(t))

   # --- Run the full pipeline ---
   signals, info = gp.egg_process(signal, sfreq)

   # signals is a DataFrame with columns: raw, filtered, phase, amplitude
   print(signals.head())

   # info is a dict of metrics
   print(f"Peak frequency : {info['peak_freq_hz']:.3f} Hz")
   print(f"Cycles detected: {info['cycle_stats']['n_cycles']}")
   print(f"Instability IC : {info['instability_coefficient']:.4f}")
   print(f"% Normogastric : {info['proportion_normogastric']:.0%}")

Step-by-Step Processing
-----------------------

You can also call each processing step individually for more control.

**1. Spectral analysis**

.. code-block:: python

   # overlap controls PSD smoothing (default 0.25; use 0.75 for Wolpert-style)
   freqs, psd = gp.psd_welch(signal, sfreq, fmin=0.01, fmax=0.1, overlap=0.75)

**2. Bandpass filtering**

.. code-block:: python

   filtered, filt_info = gp.apply_bandpass(signal, sfreq, low_hz=0.033, high_hz=0.067)

**3. Phase and amplitude extraction**

.. code-block:: python

   phase, analytic = gp.instantaneous_phase(filtered)
   amplitude = np.abs(analytic)

**4. Cycle detection**

.. code-block:: python

   durations = gp.cycle_durations(phase, t)

**5. Metric computation**

.. code-block:: python

   from gastropy.metrics import band_power, NORMOGASTRIA

   bp = band_power(freqs, psd, NORMOGASTRIA)
   ic = gp.instability_coefficient(durations)

Artifact Detection
------------------

``detect_phase_artifacts`` flags cycles with non-monotonic phase or
outlier durations, following the method of Wolpert et al. (2020).

.. code-block:: python

   artifacts = gp.detect_phase_artifacts(signals["phase"].values, t)
   print(f"Artifacts found: {artifacts['n_artifacts']}")
   print(f"Artifact mask  : {artifacts['artifact_mask'].sum()} samples flagged")

Channel Selection
-----------------

When working with multi-channel EGG data, ``select_best_channel`` ranks
channels by peak power in the normogastric band and returns the best one.
Peak detection uses ``scipy.signal.find_peaks`` for robust local-maximum
identification.

.. code-block:: python

   # data shape: (n_channels, n_samples)
   best_idx, peak_freq, freqs, psd = gp.select_best_channel(data, sfreq)
   print(f"Best channel: {best_idx} (peak at {peak_freq:.3f} Hz)")

Visualization
-------------

GastroPy provides publication-ready plots that accept raw arrays and
return ``(fig, ax)`` tuples for easy customization.

.. code-block:: python

   # PSD with normogastric band shading
   fig, ax = gp.plot_psd(freqs, psd)

   # 4-panel EGG overview (raw, filtered, phase, amplitude)
   fig, axes = gp.plot_egg_overview(signals, sfreq)

   # Cycle duration histogram with normogastric boundaries
   fig, ax = gp.plot_cycle_histogram(info["cycle_durations_s"])

   # Phase time series with artifact regions highlighted
   fig, ax = gp.plot_artifacts(signals["phase"].values, t, artifacts)

   # Comprehensive multi-panel figure (overview + optional fMRI phase)
   fig, axes = gp.plot_egg_comprehensive(signals, sfreq, artifact_info=artifacts)

fMRI-Concurrent EGG
--------------------

For EGG recorded inside an MRI scanner, ``gastropy.neuro.fmri`` provides
utilities for parsing scanner triggers, windowing data by volume, and
removing transient volumes.

.. code-block:: python

   from gastropy.neuro.fmri import (
       find_scanner_triggers,
       create_volume_windows,
       phase_per_volume,
       apply_volume_cuts,
   )

   # Parse R128 triggers from MNE annotations
   onsets = find_scanner_triggers(raw.annotations, label="R128")

   # Create per-volume windows and extract phase
   windows = create_volume_windows(onsets, tr=1.856, n_volumes=420)
   phases = phase_per_volume(analytic, windows)

   # Remove transient volumes
   phases_trimmed = apply_volume_cuts(phases, begin_cut=21, end_cut=21)

   # Visualize per-volume phase with cut regions
   fig, ax = gp.plot_volume_phase(phases, tr=1.856, cut_start=21, cut_end=21)

Time-Frequency Analysis
-----------------------

Decompose the EGG signal across gastric frequency bands (bradygastria,
normogastria, tachygastria) for per-band metrics.

.. code-block:: python

   # Single-band decomposition
   result = gp.band_decompose(signal, sfreq)

   # Multi-band analysis across all gastric bands
   results = gp.multiband_analysis(signal, sfreq)
   for band_name, res in results.items():
       print(f"{band_name}: peak={res['peak_freq_hz']:.3f} Hz, "
             f"IC={res['instability_coefficient']:.3f}")

What's Next
-----------

GastroPy is under active development. Upcoming modules include
gastric-brain phase coupling, data I/O with BIDS support, and
statistical testing. See the :doc:`API Reference <api/index>`
for the full list of currently available functions.
