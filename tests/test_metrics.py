"""Tests for gastropy.metrics module."""

import numpy as np
import pytest

from gastropy.metrics import (
    BRADYGASTRIA,
    GASTRIC_BANDS,
    NORMOGASTRIA,
    TACHYGASTRIA,
    assess_quality,
    band_power,
    cycle_stats,
    instability_coefficient,
    proportion_normogastric,
)

# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------


class TestBandDefinitions:
    def test_normogastria_range(self):
        """Normogastria should be 2-4 cpm."""
        assert NORMOGASTRIA.f_lo == pytest.approx(0.03333, rel=1e-3)
        assert NORMOGASTRIA.f_hi == pytest.approx(0.06666, rel=1e-3)
        assert NORMOGASTRIA.cpm_lo == pytest.approx(2.0, rel=1e-2)
        assert NORMOGASTRIA.cpm_hi == pytest.approx(4.0, rel=1e-2)

    def test_bradygastria_range(self):
        """Bradygastria should be 1-2 cpm."""
        assert BRADYGASTRIA.cpm_lo == pytest.approx(1.2, rel=1e-1)
        assert BRADYGASTRIA.cpm_hi == pytest.approx(1.8, rel=1e-1)

    def test_tachygastria_range(self):
        """Tachygastria should be 4-10 cpm."""
        assert TACHYGASTRIA.cpm_lo == pytest.approx(4.2, rel=1e-1)
        assert TACHYGASTRIA.cpm_hi == pytest.approx(10.2, rel=1e-1)

    def test_gastric_bands_list(self):
        """GASTRIC_BANDS should contain all three bands."""
        assert len(GASTRIC_BANDS) == 3
        names = {b.name for b in GASTRIC_BANDS}
        assert names == {"brady", "normo", "tachy"}

    def test_band_is_frozen(self):
        """GastricBand should be immutable."""
        with pytest.raises(AttributeError):
            NORMOGASTRIA.f_lo = 0.1


# ---------------------------------------------------------------------------
# band_power
# ---------------------------------------------------------------------------


class TestBandPower:
    def test_detects_peak_in_band(self):
        """band_power should find the peak frequency in the band."""
        # Synthetic PSD with a peak at 0.05 Hz (in normogastria)
        freqs = np.linspace(0.01, 0.2, 200)
        psd = np.exp(-((freqs - 0.05) ** 2) / (2 * 0.005**2))
        result = band_power(freqs, psd, NORMOGASTRIA)
        assert abs(result["peak_freq_hz"] - 0.05) < 0.005
        assert result["max_power"] > 0
        assert result["mean_power"] > 0
        assert 0 < result["prop_power"] <= 1.0

    def test_empty_band_returns_nan(self):
        """If band has no frequency coverage, all values should be NaN."""
        freqs = np.linspace(0.5, 1.0, 100)
        psd = np.ones(100)
        result = band_power(freqs, psd, NORMOGASTRIA)
        assert np.isnan(result["peak_freq_hz"])

    def test_custom_total_range(self):
        """Custom total_range should work."""
        freqs = np.linspace(0.01, 0.2, 200)
        psd = np.ones(200)
        result = band_power(freqs, psd, NORMOGASTRIA, total_range=(0.01, 0.1))
        assert not np.isnan(result["prop_power"])


# ---------------------------------------------------------------------------
# instability_coefficient
# ---------------------------------------------------------------------------


class TestInstabilityCoefficient:
    def test_stable_rhythm(self):
        """Very consistent cycle durations should give low IC."""
        durs = [20.0, 20.0, 20.0, 20.0, 20.0]
        ic = instability_coefficient(durs)
        assert ic == 0.0

    def test_variable_rhythm(self):
        """Variable durations should give a positive IC."""
        durs = [15.0, 20.0, 25.0, 18.0, 22.0]
        ic = instability_coefficient(durs)
        assert ic > 0

    def test_too_few_cycles(self):
        """Fewer than 2 cycles should return NaN."""
        assert np.isnan(instability_coefficient([20.0]))
        assert np.isnan(instability_coefficient([]))

    def test_handles_nan(self):
        """NaN values in input should be ignored."""
        durs = [20.0, np.nan, 20.0, np.nan, 20.0]
        ic = instability_coefficient(durs)
        assert ic == 0.0


# ---------------------------------------------------------------------------
# cycle_stats
# ---------------------------------------------------------------------------


class TestCycleStats:
    def test_basic_stats(self):
        """Should compute correct mean and SD."""
        durs = [20.0, 20.0, 20.0]
        stats = cycle_stats(durs)
        assert stats["n_cycles"] == 3
        assert stats["mean_cycle_dur_s"] == pytest.approx(20.0)
        assert stats["sd_cycle_dur_s"] == pytest.approx(0.0)
        assert stats["prop_within_3sd"] == 1.0

    def test_empty_returns_nan(self):
        """Empty input should return NaN values."""
        stats = cycle_stats([])
        assert stats["n_cycles"] == 0
        assert np.isnan(stats["mean_cycle_dur_s"])

    def test_3sigma_bounds(self):
        """3-sigma bounds should be mean ± 3*SD."""
        durs = [18.0, 20.0, 22.0]
        stats = cycle_stats(durs)
        assert stats["lower_3sd_s"] < stats["mean_cycle_dur_s"]
        assert stats["upper_3sd_s"] > stats["mean_cycle_dur_s"]


# ---------------------------------------------------------------------------
# proportion_normogastric
# ---------------------------------------------------------------------------


class TestProportionNormogastric:
    def test_all_normogastric(self):
        """All cycles in range should give 1.0."""
        assert proportion_normogastric([20.0, 22.0, 25.0]) == pytest.approx(1.0)

    def test_none_normogastric(self):
        """No cycles in range should give 0.0."""
        assert proportion_normogastric([5.0, 40.0, 50.0]) == pytest.approx(0.0)

    def test_mixed(self):
        """50% in range should give 0.5."""
        assert proportion_normogastric([20.0, 22.0, 10.0, 35.0]) == pytest.approx(0.5)

    def test_empty_returns_nan(self):
        """Empty input should return NaN."""
        assert np.isnan(proportion_normogastric([]))


# ---------------------------------------------------------------------------
# assess_quality
# ---------------------------------------------------------------------------


class TestAssessQuality:
    def test_good_quality(self):
        """Good recording should pass all checks."""
        qc = assess_quality(n_cycles=15, cycle_sd=3.0, prop_normo=0.8)
        assert qc["sufficient_cycles"] is True
        assert qc["stable_rhythm"] is True
        assert qc["normogastric_dominant"] is True
        assert qc["overall"] is True

    def test_too_few_cycles(self):
        """Too few cycles should fail overall."""
        qc = assess_quality(n_cycles=5, cycle_sd=3.0, prop_normo=0.8)
        assert qc["sufficient_cycles"] is False
        assert qc["overall"] is False

    def test_unstable_but_normogastric(self):
        """High SD but good normogastric proportion should still pass."""
        qc = assess_quality(n_cycles=15, cycle_sd=8.0, prop_normo=0.75)
        assert qc["stable_rhythm"] is False
        assert qc["normogastric_dominant"] is True
        assert qc["overall"] is True

    def test_stable_but_not_normogastric(self):
        """Low SD but poor normogastric proportion should still pass."""
        qc = assess_quality(n_cycles=15, cycle_sd=3.0, prop_normo=0.3)
        assert qc["stable_rhythm"] is True
        assert qc["normogastric_dominant"] is False
        assert qc["overall"] is True

    def test_bad_quality(self):
        """Unstable AND non-normogastric should fail."""
        qc = assess_quality(n_cycles=15, cycle_sd=8.0, prop_normo=0.3)
        assert qc["overall"] is False
