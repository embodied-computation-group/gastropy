"""Tests for gastropy.coupling module."""

import numpy as np
import pytest

from gastropy.coupling import (
    circular_mean,
    coupling_zscore,
    phase_locking_value,
    phase_locking_value_complex,
    rayleigh_test,
    resultant_length,
    surrogate_plv,
)


def _make_phase_signals(freq_a=0.05, freq_b=0.05, sfreq=10.0, duration=300.0, phase_offset=0.0, noise=0.0, seed=42):
    """Create synthetic phase signals for testing."""
    t = np.arange(0, duration, 1.0 / sfreq)
    phase_a = 2 * np.pi * freq_a * t
    phase_b = 2 * np.pi * freq_b * t + phase_offset
    if noise > 0:
        rng = np.random.default_rng(seed)
        phase_a += noise * rng.standard_normal(len(t))
        phase_b += noise * rng.standard_normal(len(t))
    return phase_a, phase_b


# ---------------------------------------------------------------------------
# phase_locking_value
# ---------------------------------------------------------------------------


class TestPhaseLockingValue:
    def test_identical_phases_give_plv_one(self):
        """Identical phase signals should have PLV = 1."""
        phase_a, _ = _make_phase_signals()
        plv = phase_locking_value(phase_a, phase_a)
        assert abs(plv - 1.0) < 1e-10

    def test_constant_offset_gives_plv_one(self):
        """Constant phase offset should give PLV = 1."""
        phase_a, phase_b = _make_phase_signals(phase_offset=np.pi / 4)
        plv = phase_locking_value(phase_a, phase_b)
        assert abs(plv - 1.0) < 1e-10

    def test_uniform_random_phases_give_low_plv(self):
        """Uncoupled random phases should give PLV near 0."""
        rng = np.random.default_rng(42)
        phase_a = rng.uniform(-np.pi, np.pi, 5000)
        phase_b = rng.uniform(-np.pi, np.pi, 5000)
        plv = phase_locking_value(phase_a, phase_b)
        assert plv < 0.05

    def test_plv_range(self):
        """PLV should be in [0, 1]."""
        phase_a, phase_b = _make_phase_signals(noise=1.0)
        plv = phase_locking_value(phase_a, phase_b)
        assert 0.0 <= plv <= 1.0

    def test_2d_phase_a(self):
        """Should compute PLV for each column of 2D phase_a."""
        rng = np.random.default_rng(42)
        n_time = 500
        n_signals = 10
        phase_b = rng.uniform(-np.pi, np.pi, n_time)

        # First column perfectly locked, rest random
        phase_a = rng.uniform(-np.pi, np.pi, (n_time, n_signals))
        phase_a[:, 0] = phase_b + 0.3  # constant offset

        plv = phase_locking_value(phase_a, phase_b)
        assert plv.shape == (n_signals,)
        assert abs(plv[0] - 1.0) < 1e-10
        assert all(plv[1:] < 0.2)

    def test_mismatched_lengths_raises(self):
        """Should raise ValueError for mismatched lengths."""
        with pytest.raises(ValueError, match="same number of timepoints"):
            phase_locking_value(np.zeros(100), np.zeros(50))

    def test_returns_scalar_for_1d(self):
        """Should return a Python float for 1D inputs."""
        plv = phase_locking_value(np.zeros(100), np.zeros(100))
        assert isinstance(plv, float)


# ---------------------------------------------------------------------------
# phase_locking_value_complex
# ---------------------------------------------------------------------------


class TestPhaseLockingValueComplex:
    def test_magnitude_equals_plv(self):
        """Magnitude of complex PLV should equal PLV."""
        phase_a, phase_b = _make_phase_signals(noise=0.5)
        plv = phase_locking_value(phase_a, phase_b)
        cplv = phase_locking_value_complex(phase_a, phase_b)
        assert abs(abs(cplv) - plv) < 1e-10

    def test_phase_lag_for_constant_offset(self):
        """Complex PLV angle should reflect constant phase offset."""
        offset = 0.7
        phase_a, phase_b = _make_phase_signals(phase_offset=offset)
        cplv = phase_locking_value_complex(phase_a, phase_b)
        # phase_a - phase_b = -offset (since phase_b = phase_a + offset)
        assert abs(np.angle(cplv) - (-offset)) < 1e-6

    def test_returns_complex(self):
        """Should return a complex number for 1D inputs."""
        cplv = phase_locking_value_complex(np.zeros(100), np.zeros(100))
        assert isinstance(cplv, complex)

    def test_2d_returns_complex_array(self):
        """Should return complex array for 2D phase_a."""
        phase_a = np.zeros((100, 5))
        phase_b = np.zeros(100)
        cplv = phase_locking_value_complex(phase_a, phase_b)
        assert cplv.shape == (5,)
        assert np.issubdtype(cplv.dtype, np.complexfloating)


# ---------------------------------------------------------------------------
# circular_mean
# ---------------------------------------------------------------------------


class TestCircularMean:
    def test_zero_phases(self):
        """Mean of zero phases should be zero."""
        assert circular_mean(np.zeros(100)) == 0.0

    def test_known_value(self):
        """Should compute correct mean for simple case."""
        # Two phases at 0 and pi/2 -> mean at pi/4
        phases = np.array([0.0, np.pi / 2])
        mean = circular_mean(phases)
        assert abs(mean - np.pi / 4) < 1e-10

    def test_wrapping(self):
        """Mean should handle wrapping around +/- pi."""
        # Phases clustered near pi/-pi should give mean near pi
        phases = np.array([np.pi - 0.1, -np.pi + 0.1])
        mean = circular_mean(phases)
        assert abs(abs(mean) - np.pi) < 0.11

    def test_returns_float(self):
        """Should return a float."""
        assert isinstance(circular_mean(np.array([0.0, 1.0])), float)


# ---------------------------------------------------------------------------
# resultant_length
# ---------------------------------------------------------------------------


class TestResultantLength:
    def test_identical_phases_give_one(self):
        """All identical phases should give R = 1."""
        assert resultant_length(np.zeros(100)) == 1.0

    def test_opposite_phases_give_zero(self):
        """Equal numbers of opposite phases should give R ~ 0."""
        phases = np.array([0.0, np.pi] * 50)
        R = resultant_length(phases)
        assert R < 1e-10

    def test_uniform_gives_low_r(self):
        """Uniform phases should give R near 0."""
        phases = np.linspace(-np.pi, np.pi, 1000, endpoint=False)
        R = resultant_length(phases)
        assert R < 0.01

    def test_range(self):
        """R should be in [0, 1]."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, 100)
        R = resultant_length(phases)
        assert 0.0 <= R <= 1.0


# ---------------------------------------------------------------------------
# rayleigh_test
# ---------------------------------------------------------------------------


class TestRayleighTest:
    def test_uniform_not_significant(self):
        """Uniform phases should not be significant."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, 500)
        z, p = rayleigh_test(phases)
        assert p > 0.05

    def test_concentrated_is_significant(self):
        """Concentrated phases should be significant."""
        rng = np.random.default_rng(42)
        phases = rng.normal(0, 0.1, 100)  # tight cluster around 0
        z, p = rayleigh_test(phases)
        assert p < 0.001

    def test_z_stat_formula(self):
        """z should equal n * R^2."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, 50)
        z, _ = rayleigh_test(phases)
        R = resultant_length(phases)
        expected_z = 50 * R**2
        assert abs(z - expected_z) < 1e-10

    def test_empty_input(self):
        """Empty input should return z=0, p=1."""
        z, p = rayleigh_test(np.array([]))
        assert z == 0.0
        assert p == 1.0

    def test_p_value_range(self):
        """p-value should be in [0, 1]."""
        rng = np.random.default_rng(42)
        phases = rng.uniform(-np.pi, np.pi, 100)
        _, p = rayleigh_test(phases)
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# surrogate_plv
# ---------------------------------------------------------------------------


class TestSurrogatePlv:
    def test_surrogate_lower_than_locked(self):
        """Surrogate PLV should be lower than empirical for locked signals."""
        phase_a, phase_b = _make_phase_signals(phase_offset=0.3)
        empirical = phase_locking_value(phase_a, phase_b)
        surr = surrogate_plv(phase_a, phase_b, seed=42)
        assert surr < empirical

    def test_surrogate_similar_for_random(self):
        """Surrogate PLV should be similar to empirical for random signals."""
        rng = np.random.default_rng(42)
        phase_a = rng.uniform(-np.pi, np.pi, 500)
        phase_b = rng.uniform(-np.pi, np.pi, 500)
        empirical = phase_locking_value(phase_a, phase_b)
        surr = surrogate_plv(phase_a, phase_b, seed=42)
        # Both should be near zero
        assert abs(empirical - surr) < 0.1

    def test_stat_all_returns_distribution(self):
        """stat='all' should return full distribution."""
        rng = np.random.default_rng(42)
        phase_a = rng.uniform(-np.pi, np.pi, 100)
        phase_b = rng.uniform(-np.pi, np.pi, 100)
        surr = surrogate_plv(phase_a, phase_b, stat="all", seed=42)
        assert surr.ndim == 1
        assert len(surr) > 1  # multiple shifts

    def test_n_surrogates_limits_count(self):
        """n_surrogates should limit the number of shifts."""
        rng = np.random.default_rng(42)
        phase_a = rng.uniform(-np.pi, np.pi, 200)
        phase_b = rng.uniform(-np.pi, np.pi, 200)
        surr = surrogate_plv(phase_a, phase_b, n_surrogates=20, stat="all", seed=42)
        assert len(surr) == 20

    def test_stat_mean(self):
        """stat='mean' should return a scalar."""
        rng = np.random.default_rng(42)
        phase_a = rng.uniform(-np.pi, np.pi, 200)
        phase_b = rng.uniform(-np.pi, np.pi, 200)
        surr = surrogate_plv(phase_a, phase_b, stat="mean", seed=42)
        assert isinstance(surr, (float, np.floating))

    def test_invalid_stat_raises(self):
        """Invalid stat should raise ValueError."""
        with pytest.raises(ValueError, match="stat must be"):
            surrogate_plv(np.zeros(100), np.zeros(100), stat="invalid")

    def test_buffer_too_large_raises(self):
        """Buffer larger than half the signal should raise."""
        with pytest.raises(ValueError, match="too large"):
            surrogate_plv(np.zeros(100), np.zeros(100), buffer_samples=60)

    def test_2d_phase_a(self):
        """Should work with 2D phase_a."""
        rng = np.random.default_rng(42)
        phase_a = rng.uniform(-np.pi, np.pi, (200, 5))
        phase_b = rng.uniform(-np.pi, np.pi, 200)
        surr = surrogate_plv(phase_a, phase_b, n_surrogates=10, seed=42)
        assert surr.shape == (5,)

    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""
        rng = np.random.default_rng(42)
        phase_a = rng.uniform(-np.pi, np.pi, 200)
        phase_b = rng.uniform(-np.pi, np.pi, 200)
        surr1 = surrogate_plv(phase_a, phase_b, n_surrogates=20, seed=123)
        surr2 = surrogate_plv(phase_a, phase_b, n_surrogates=20, seed=123)
        assert surr1 == surr2


# ---------------------------------------------------------------------------
# coupling_zscore
# ---------------------------------------------------------------------------


class TestCouplingZscore:
    def test_simple_difference(self):
        """With scalar surrogate, should return simple difference."""
        z = coupling_zscore(0.3, 0.15)
        assert abs(z - 0.15) < 1e-10

    def test_zscore_with_distribution(self):
        """With distribution surrogate, should return z-score."""
        rng = np.random.default_rng(42)
        surr_dist = rng.normal(0.1, 0.02, 1000)
        z = coupling_zscore(0.2, surr_dist)
        # z should be approximately (0.2 - 0.1) / 0.02 = 5.0
        assert abs(z - 5.0) < 0.5

    def test_array_inputs(self):
        """Should work with array empirical and array surrogate."""
        emp = np.array([0.3, 0.5])
        surr = np.array([0.1, 0.2])
        z = coupling_zscore(emp, surr)
        np.testing.assert_allclose(z, [0.2, 0.3])


# ---------------------------------------------------------------------------
# Top-level import test
# ---------------------------------------------------------------------------


class TestCouplingImport:
    def test_flat_namespace(self):
        """Coupling functions should be accessible from gastropy.*."""
        import gastropy

        assert hasattr(gastropy, "phase_locking_value")
        assert hasattr(gastropy, "phase_locking_value_complex")
        assert hasattr(gastropy, "surrogate_plv")
        assert hasattr(gastropy, "coupling_zscore")
        assert hasattr(gastropy, "circular_mean")
        assert hasattr(gastropy, "resultant_length")
        assert hasattr(gastropy, "rayleigh_test")
