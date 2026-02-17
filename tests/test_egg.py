"""Tests for gastropy.egg module."""

import numpy as np
import pandas as pd
import pytest

from gastropy.egg import egg_clean, egg_process, egg_process_multichannel, select_best_channel, select_peak_frequency

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


# ---------------------------------------------------------------------------
# egg_process_multichannel
# ---------------------------------------------------------------------------


def _make_multichannel_egg(n_channels=4, freq=0.05, sfreq=10.0, duration=300.0, noise=0.1, seed=0):
    """Create multi-channel EGG with a shared gastric component."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / sfreq)
    gastric = np.sin(2 * np.pi * freq * t)
    return np.stack([gastric + noise * rng.standard_normal(len(t)) for _ in range(n_channels)])


class TestEggProcessMultichannel:
    """Tests for egg_process_multichannel."""

    def test_per_channel_returns_expected_keys(self):
        """per_channel result should have channels, best_idx, summary, method."""
        data = _make_multichannel_egg()
        result = egg_process_multichannel(data, sfreq=10.0, method="per_channel")
        for key in ("channels", "best_idx", "summary", "method"):
            assert key in result

    def test_per_channel_channels_dict_length(self):
        """channels dict should have one entry per channel."""
        n = 3
        data = _make_multichannel_egg(n_channels=n)
        result = egg_process_multichannel(data, sfreq=10.0)
        assert len(result["channels"]) == n

    def test_per_channel_signals_df_type(self):
        """Each channel value should be a (DataFrame, dict) tuple."""
        data = _make_multichannel_egg(n_channels=3)
        result = egg_process_multichannel(data, sfreq=10.0)
        for ch_idx, (signals_df, info) in result["channels"].items():
            assert isinstance(signals_df, pd.DataFrame)
            assert isinstance(info, dict)

    def test_per_channel_summary_shape(self):
        """Summary DataFrame should have one row per channel."""
        n = 4
        data = _make_multichannel_egg(n_channels=n)
        result = egg_process_multichannel(data, sfreq=10.0)
        assert len(result["summary"]) == n

    def test_per_channel_method_label(self):
        """method field should be 'per_channel'."""
        data = _make_multichannel_egg()
        result = egg_process_multichannel(data, sfreq=10.0, method="per_channel")
        assert result["method"] == "per_channel"

    def test_best_channel_returns_dataframe(self):
        """best_channel result should contain a signals DataFrame."""
        data = _make_multichannel_egg()
        result = egg_process_multichannel(data, sfreq=10.0, method="best_channel")
        assert "signals" in result
        assert isinstance(result["signals"], pd.DataFrame)

    def test_best_channel_info_has_best_idx(self):
        """best_channel info should include best_channel_idx."""
        data = _make_multichannel_egg()
        result = egg_process_multichannel(data, sfreq=10.0, method="best_channel")
        assert "best_channel_idx" in result["info"]

    def test_best_channel_signals_columns(self):
        """best_channel signals DataFrame should have the standard columns."""
        data = _make_multichannel_egg()
        result = egg_process_multichannel(data, sfreq=10.0, method="best_channel")
        for col in ("raw", "filtered", "phase", "amplitude"):
            assert col in result["signals"].columns

    def test_ica_method_runs(self):
        """ica method should complete without error on a clear gastric signal."""
        data = _make_multichannel_egg(n_channels=4, noise=0.1)
        result = egg_process_multichannel(data, sfreq=10.0, method="ica", ica_snr_threshold=1.5)
        assert result["method"] == "ica"
        assert "ica_info" in result

    def test_ica_shape_preserved(self):
        """ica method output channels should match input channel count."""
        n = 4
        data = _make_multichannel_egg(n_channels=n, noise=0.1)
        result = egg_process_multichannel(data, sfreq=10.0, method="ica", ica_snr_threshold=1.5)
        assert len(result["channels"]) == n

    def test_raises_on_1d_input(self):
        """Should raise ValueError when given a 1D array."""
        _, sig = _make_egg_signal()
        with pytest.raises(ValueError, match="2D"):
            egg_process_multichannel(sig, sfreq=10.0)

    def test_raises_on_single_channel(self):
        """Should raise ValueError when given only 1 channel."""
        _, sig = _make_egg_signal()
        data = sig[np.newaxis, :]  # (1, n_samples)
        with pytest.raises(ValueError, match="at least 2"):
            egg_process_multichannel(data, sfreq=10.0)

    def test_raises_on_unknown_method(self):
        """Should raise ValueError for an unsupported method name."""
        data = _make_multichannel_egg()
        with pytest.raises(ValueError, match="Unknown method"):
            egg_process_multichannel(data, sfreq=10.0, method="bad_method")
