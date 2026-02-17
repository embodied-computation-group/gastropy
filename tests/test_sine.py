"""Tests for gastropy.signal.sine — sine wave fitting and evaluation."""

import numpy as np
import pytest

from gastropy.signal import fit_sine, sine_model


def _make_sine(freq=0.05, phase=0.0, amp=1.0, sfreq=10.0, duration=300.0):
    """Create a clean sinusoidal signal with known parameters."""
    t = np.arange(0, duration, 1.0 / sfreq)
    return t, amp * np.sin(2 * np.pi * freq * t + phase)


# ---------------------------------------------------------------------------
# sine_model
# ---------------------------------------------------------------------------


class TestSineModel:
    def test_output_shape(self):
        """Output shape should match input time array."""
        t = np.linspace(0, 10, 200)
        y = sine_model(t, freq=0.05, phase=0.0, amp=1.0)
        assert y.shape == t.shape

    def test_known_values(self):
        """Output should match hand-computed sine values."""
        t = np.array([0.0, 5.0, 10.0])  # 0.05 Hz → period 20 s
        y = sine_model(t, freq=0.05, phase=0.0, amp=1.0)
        expected = np.sin(2 * np.pi * 0.05 * t)
        np.testing.assert_allclose(y, expected)

    def test_amplitude_scaling(self):
        """Peak absolute value should equal the amplitude."""
        t = np.linspace(0, 100, 10000)
        y = sine_model(t, freq=0.05, phase=0.0, amp=3.7)
        assert abs(np.max(np.abs(y)) - 3.7) < 1e-6

    def test_phase_shift(self):
        """A phase shift of π/2 should match cos."""
        t = np.linspace(0, 100, 10000)
        y = sine_model(t, freq=0.05, phase=np.pi / 2, amp=1.0)
        expected = np.cos(2 * np.pi * 0.05 * t)
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_returns_ndarray(self):
        """Output should be a numpy array."""
        t = np.linspace(0, 10, 100)
        y = sine_model(t, freq=0.05, phase=0.0, amp=1.0)
        assert isinstance(y, np.ndarray)

    def test_accepts_list_input(self):
        """Should accept Python lists as time input."""
        t = list(range(100))
        y = sine_model(t, freq=0.05, phase=0.0, amp=1.0)
        assert len(y) == 100


# ---------------------------------------------------------------------------
# fit_sine
# ---------------------------------------------------------------------------


class TestFitSine:
    def test_recovers_amplitude_fixed_freq(self):
        """Fitted amplitude should be close to the true amplitude (fixed freq)."""
        _, sig = _make_sine(freq=0.05, amp=2.5, phase=0.3)
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        assert abs(abs(result["amplitude"]) - 2.5) < 0.1

    def test_recovers_phase_fixed_freq(self):
        """Fitted phase should be close to the true phase (fixed freq)."""
        _, sig = _make_sine(freq=0.05, amp=1.0, phase=0.7)
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        # Phase is only meaningful mod π when amplitude can be negative;
        # check that reconstructed signal matches original instead.
        t = np.arange(0, 300, 0.1)
        y_fit = sine_model(t, result["freq_hz"], result["phase_rad"], result["amplitude"])
        np.testing.assert_allclose(y_fit, sig, atol=1e-3)

    def test_recovers_frequency_free_fit(self):
        """Fitted frequency should be close to true frequency when freq=None."""
        _, sig = _make_sine(freq=0.05, amp=1.0, phase=0.0)
        result = fit_sine(sig, sfreq=10.0, freq=None)
        assert abs(result["freq_hz"] - 0.05) < 0.005

    def test_fixed_freq_passthrough(self):
        """freq_hz in result should equal the provided fixed freq."""
        _, sig = _make_sine(freq=0.05)
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        assert result["freq_hz"] == pytest.approx(0.05)

    def test_result_keys(self):
        """Result dict should contain the four expected keys."""
        _, sig = _make_sine()
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        for key in ("freq_hz", "phase_rad", "amplitude", "residual"):
            assert key in result, f"Missing key: {key}"

    def test_residual_is_finite(self):
        """Residual should be a finite non-negative float."""
        _, sig = _make_sine()
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        assert np.isfinite(result["residual"])
        assert result["residual"] >= 0.0

    def test_residual_near_zero_for_clean_signal(self):
        """Residual should be near zero for a perfect sinusoid."""
        _, sig = _make_sine(freq=0.05, amp=1.0, phase=0.0)
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        assert result["residual"] < 1e-6

    def test_freq_init_used_for_free_fit(self):
        """freq_init should guide the free-frequency fit away from default 0.05 Hz."""
        _, sig = _make_sine(freq=0.1, amp=1.0, phase=0.0, duration=200.0)
        # With the correct freq_init, the optimizer finds the right frequency
        result = fit_sine(sig, sfreq=10.0, freq=None, freq_init=0.1)
        assert abs(result["freq_hz"] - 0.1) < 0.01

    def test_returns_dict(self):
        """Return type should be a plain dict."""
        _, sig = _make_sine()
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        assert isinstance(result, dict)

    def test_all_result_values_are_floats(self):
        """All scalar result values should be Python floats."""
        _, sig = _make_sine()
        result = fit_sine(sig, sfreq=10.0, freq=0.05)
        for key in ("freq_hz", "phase_rad", "amplitude", "residual"):
            assert isinstance(result[key], float), f"{key} is not a float"
