"""Tests for gastropy.egg module."""

import numpy as np
import pandas as pd

from gastropy.egg import egg_clean, egg_process, select_best_channel, select_peak_frequency

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_egg_signal(freq_hz=0.05, sfreq=10.0, duration=300.0, noise=0.1):
    """Create a synthetic EGG signal (3 cpm by default)."""
    rng = np.random.default_rng(42)
    t = np.arange(0, duration, 1.0 / sfreq)
    sig = np.sin(2 * np.pi * freq_hz * t) + noise * rng.standard_normal(len(t))
    return t, sig


# ---------------------------------------------------------------------------
# select_best_channel
# ---------------------------------------------------------------------------


class TestSelectBestChannel:
    def test_finds_channel_with_gastric_peak(self):
        """Should select the channel containing the gastric rhythm."""
        rng = np.random.default_rng(42)
        sfreq = 10.0
        t = np.arange(0, 300, 1.0 / sfreq)

        # Channel 0: noise only
        ch0 = rng.standard_normal(len(t))
        # Channel 1: strong 3 cpm signal
        ch1 = 5.0 * np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.standard_normal(len(t))
        # Channel 2: noise only
        ch2 = rng.standard_normal(len(t))

        data = np.array([ch0, ch1, ch2])
        best_idx, peak_freq, _, _ = select_best_channel(data, sfreq)
        assert best_idx == 1
        assert abs(peak_freq - 0.05) < 0.01

    def test_single_channel(self):
        """Should work with single-channel data."""
        _, sig = _make_egg_signal()
        best_idx, peak_freq, freqs, psd = select_best_channel(sig.reshape(1, -1), sfreq=10.0)
        assert best_idx == 0
        assert len(freqs) > 0


# ---------------------------------------------------------------------------
# select_peak_frequency
# ---------------------------------------------------------------------------


class TestSelectPeakFrequency:
    def test_finds_known_frequency(self):
        """Should find the peak at the signal frequency."""
        _, sig = _make_egg_signal(freq_hz=0.05)
        peak_freq, freqs, psd = select_peak_frequency(sig, sfreq=10.0)
        assert abs(peak_freq - 0.05) < 0.005

    def test_finds_peak_not_edge(self):
        """find_peaks should select a true local maximum, not a band edge."""
        rng = np.random.default_rng(99)
        sfreq = 10.0
        t = np.arange(0, 300, 1.0 / sfreq)
        # Signal with peak at 0.05 Hz plus weaker peak at 0.04 Hz
        sig = 3.0 * np.sin(2 * np.pi * 0.05 * t) + 1.0 * np.sin(2 * np.pi * 0.04 * t)
        sig += 0.1 * rng.standard_normal(len(t))
        peak_freq, _, _ = select_peak_frequency(sig, sfreq=sfreq)
        # Should pick the dominant peak at 0.05, not 0.04 or band edges
        assert abs(peak_freq - 0.05) < 0.005


# ---------------------------------------------------------------------------
# egg_clean
# ---------------------------------------------------------------------------


class TestEggClean:
    def test_output_same_length(self):
        """Cleaned signal should have the same length as input."""
        _, sig = _make_egg_signal()
        cleaned, info = egg_clean(sig, sfreq=10.0)
        assert len(cleaned) == len(sig)

    def test_custom_frequency_bounds(self):
        """Should accept custom low_hz and high_hz."""
        _, sig = _make_egg_signal()
        cleaned, info = egg_clean(sig, sfreq=10.0, low_hz=0.02, high_hz=0.08)
        assert len(cleaned) == len(sig)


# ---------------------------------------------------------------------------
# egg_process
# ---------------------------------------------------------------------------


class TestEggProcess:
    def test_returns_dataframe_and_dict(self):
        """Should return a DataFrame and info dict."""
        _, sig = _make_egg_signal()
        signals, info = egg_process(sig, sfreq=10.0)
        assert isinstance(signals, pd.DataFrame)
        assert isinstance(info, dict)

    def test_dataframe_columns(self):
        """DataFrame should have expected columns."""
        _, sig = _make_egg_signal()
        signals, _ = egg_process(sig, sfreq=10.0)
        assert set(signals.columns) == {"raw", "filtered", "phase", "amplitude"}

    def test_info_contains_metrics(self):
        """Info dict should contain all expected metric keys."""
        _, sig = _make_egg_signal()
        _, info = egg_process(sig, sfreq=10.0)
        assert "peak_freq_hz" in info
        assert "cycle_durations_s" in info
        assert "cycle_stats" in info
        assert "instability_coefficient" in info
        assert "proportion_normogastric" in info
        assert "band_power" in info
        assert "filter" in info

    def test_detects_cycles(self):
        """Should detect cycles in a clean sinusoidal signal."""
        _, sig = _make_egg_signal(freq_hz=0.05, noise=0.05, duration=300.0)
        _, info = egg_process(sig, sfreq=10.0)
        durs = info["cycle_durations_s"]
        assert len(durs) > 5
        assert abs(np.mean(durs) - 20.0) < 2.0

    def test_peak_frequency_is_correct(self):
        """Peak frequency should be close to the signal frequency."""
        _, sig = _make_egg_signal(freq_hz=0.05, noise=0.05)
        _, info = egg_process(sig, sfreq=10.0)
        assert abs(info["peak_freq_hz"] - 0.05) < 0.005
