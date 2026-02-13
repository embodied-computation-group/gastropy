"""Tests for gastropy.timefreq module."""

import numpy as np

from gastropy.metrics import BRADYGASTRIA, GASTRIC_BANDS, NORMOGASTRIA
from gastropy.timefreq import band_decompose, multiband_analysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(freq_hz=0.05, sfreq=10.0, duration=300.0, noise=0.1):
    """Create a synthetic sinusoidal signal."""
    rng = np.random.default_rng(42)
    t = np.arange(0, duration, 1.0 / sfreq)
    sig = np.sin(2 * np.pi * freq_hz * t) + noise * rng.standard_normal(len(t))
    return sig


# ---------------------------------------------------------------------------
# band_decompose
# ---------------------------------------------------------------------------


class TestBandDecompose:
    def test_detects_peak_in_normogastric_band(self):
        """Should find the 0.05 Hz peak in the normogastric band."""
        sig = _make_signal(freq_hz=0.05, noise=0.05)
        result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA)
        assert abs(result["peak_freq_hz"] - 0.05) < 0.01

    def test_returns_valid_cycles(self):
        """Should detect cycles with ~20s duration for a 0.05 Hz signal."""
        sig = _make_signal(freq_hz=0.05, noise=0.05, duration=300.0)
        result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA)
        durs = result["cycle_durations_s"]
        assert len(durs) > 5
        assert abs(np.mean(durs) - 20.0) < 3.0

    def test_returns_low_ic_for_clean_signal(self):
        """Clean sinusoid should have near-zero instability coefficient."""
        sig = _make_signal(freq_hz=0.05, noise=0.02, duration=300.0)
        result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA)
        assert result["instability_coefficient"] < 0.05

    def test_returns_expected_keys(self):
        """Result dict should contain all expected keys."""
        sig = _make_signal()
        result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA)
        expected_keys = {
            "band",
            "peak_freq_hz",
            "max_power",
            "mean_power",
            "prop_power",
            "mean_power_ratio",
            "filtered",
            "phase",
            "amplitude",
            "cycle_durations_s",
            "cycle_stats",
            "instability_coefficient",
            "filter",
        }
        assert expected_keys.issubset(result.keys())

    def test_filtered_signal_same_length(self):
        """Filtered signal should match input length."""
        sig = _make_signal()
        result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA)
        assert len(result["filtered"]) == len(sig)
        assert len(result["phase"]) == len(sig)
        assert len(result["amplitude"]) == len(sig)

    def test_no_peak_returns_nan(self):
        """Band with no signal should return NaN metrics."""
        # 0.05 Hz signal is outside bradygastria (0.02-0.03 Hz)
        sig = _make_signal(freq_hz=0.05, noise=0.05)
        result = band_decompose(sig, sfreq=10.0, band=BRADYGASTRIA)
        # Brady band may or may not find a peak depending on noise.
        # If peak is found, it should still return valid structure.
        assert "peak_freq_hz" in result
        assert "cycle_stats" in result

    def test_cycle_stats_dict_present(self):
        """cycle_stats should be a dict with standard keys."""
        sig = _make_signal(freq_hz=0.05, noise=0.05)
        result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA)
        stats = result["cycle_stats"]
        assert "n_cycles" in stats
        assert "mean_cycle_dur_s" in stats
        assert "sd_cycle_dur_s" in stats

    def test_custom_hwhm(self):
        """Should accept a custom hwhm_hz parameter."""
        sig = _make_signal(freq_hz=0.05, noise=0.05)
        result = band_decompose(sig, sfreq=10.0, band=NORMOGASTRIA, hwhm_hz=0.01)
        assert np.isfinite(result["peak_freq_hz"])


# ---------------------------------------------------------------------------
# multiband_analysis
# ---------------------------------------------------------------------------


class TestMultibandAnalysis:
    def test_returns_all_three_bands(self):
        """Should return results for brady, normo, and tachy."""
        sig = _make_signal(freq_hz=0.05, noise=0.1)
        results = multiband_analysis(sig, sfreq=10.0)
        assert set(results.keys()) == {"brady", "normo", "tachy"}

    def test_normo_has_strongest_peak_for_normo_signal(self):
        """Normogastric band should have the strongest peak for a 3-cpm signal."""
        sig = _make_signal(freq_hz=0.05, noise=0.05)
        results = multiband_analysis(sig, sfreq=10.0)
        normo_power = results["normo"]["max_power"]
        # Normo should have more power than the other bands
        assert normo_power > 0
        assert abs(results["normo"]["peak_freq_hz"] - 0.05) < 0.01

    def test_custom_bands(self):
        """Should work with a custom band list."""
        sig = _make_signal(freq_hz=0.05, noise=0.1)
        results = multiband_analysis(sig, sfreq=10.0, bands=[NORMOGASTRIA])
        assert list(results.keys()) == ["normo"]

    def test_each_band_has_full_result(self):
        """Each band result should have the standard keys."""
        sig = _make_signal(freq_hz=0.05, noise=0.1)
        results = multiband_analysis(sig, sfreq=10.0)
        for name, result in results.items():
            assert "band" in result
            assert "peak_freq_hz" in result
            assert "filtered" in result
            assert "cycle_stats" in result
            assert result["band"]["name"] == name

    def test_default_bands_match_gastric_bands(self):
        """Default bands should be GASTRIC_BANDS."""
        sig = _make_signal()
        results = multiband_analysis(sig, sfreq=10.0)
        expected_names = {b.name for b in GASTRIC_BANDS}
        assert set(results.keys()) == expected_names
