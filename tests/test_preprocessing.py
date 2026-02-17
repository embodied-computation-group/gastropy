"""Tests for gastropy.signal.preprocessing — time-domain artifact removal."""

import numpy as np
import pytest

from gastropy.signal import hampel_filter, mad_filter, remove_movement_artifacts


def _make_sinusoid(freq=0.05, sfreq=10.0, duration=300.0, amplitude=1.0, seed=0):
    """Create a clean gastric sinusoid."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / sfreq)
    return np.sin(2 * np.pi * freq * t) * amplitude + 0.05 * rng.standard_normal(len(t))


# ---
# hampel_filter
# ---


class TestHampelFilter:
    """Tests for hampel_filter."""

    def test_spike_removed_1d(self):
        """A large isolated spike should be replaced by the local median."""
        sig = _make_sinusoid()
        sig[100] = 500.0  # inject spike
        cleaned = hampel_filter(sig)
        assert np.abs(cleaned[100]) < 5.0, "Spike not removed"

    def test_shape_preserved_1d(self):
        """Output shape must match input for 1D arrays."""
        sig = _make_sinusoid()
        cleaned = hampel_filter(sig)
        assert cleaned.shape == sig.shape

    def test_shape_preserved_2d(self):
        """Output shape must match input for 2D multi-channel arrays."""
        data = np.stack([_make_sinusoid(seed=i) for i in range(4)])
        cleaned = hampel_filter(data)
        assert cleaned.shape == data.shape

    def test_spike_removed_2d(self):
        """Spikes in individual channels of a 2D array should be removed."""
        data = np.stack([_make_sinusoid(seed=i) for i in range(3)])
        data[1, 50] = 500.0  # spike in channel 1
        cleaned = hampel_filter(data)
        assert np.abs(cleaned[1, 50]) < 5.0

    def test_clean_signal_unchanged(self):
        """A spike-free signal should be largely unaffected."""
        sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 0.1))
        cleaned = hampel_filter(sig)
        assert np.allclose(sig, cleaned, atol=1e-6)

    def test_returns_ndarray(self):
        """Output should always be a numpy array."""
        sig = _make_sinusoid()
        cleaned = hampel_filter(sig)
        assert isinstance(cleaned, np.ndarray)

    def test_k_parameter_effect(self):
        """Larger k (wider window) should handle broader spike clusters."""
        sig = np.zeros(200)
        sig[50:55] = 100.0  # 5-sample plateau spike
        cleaned = hampel_filter(sig, k=5)
        assert np.all(np.abs(cleaned[50:55]) < 10.0)


# ---
# mad_filter
# ---


class TestMadFilter:
    """Tests for mad_filter."""

    def test_global_outlier_removed_1d(self):
        """A large global outlier should be replaced by the global median."""
        rng = np.random.default_rng(1)
        sig = rng.standard_normal(1000)
        sig[200] = 100.0  # large outlier
        cleaned = mad_filter(sig)
        assert np.abs(cleaned[200]) < 5.0

    def test_shape_preserved_1d(self):
        """Output shape must match input for 1D arrays."""
        sig = _make_sinusoid()
        cleaned = mad_filter(sig)
        assert cleaned.shape == sig.shape

    def test_shape_preserved_2d(self):
        """Output shape must match input for 2D arrays."""
        data = np.stack([_make_sinusoid(seed=i) for i in range(3)])
        cleaned = mad_filter(data)
        assert cleaned.shape == data.shape

    def test_outlier_per_channel_2d(self):
        """Each channel's outlier should be cleaned independently."""
        data = np.stack([_make_sinusoid(seed=i) for i in range(3)])
        data[0, 10] = 200.0
        data[2, 50] = -200.0
        cleaned = mad_filter(data)
        assert np.abs(cleaned[0, 10]) < 10.0
        assert np.abs(cleaned[2, 50]) < 10.0

    def test_returns_ndarray(self):
        """Output should always be a numpy array."""
        cleaned = mad_filter(_make_sinusoid())
        assert isinstance(cleaned, np.ndarray)

    def test_moderate_values_unchanged(self):
        """Samples within the threshold should not be replaced."""
        sig = np.zeros(100)
        cleaned = mad_filter(sig)
        # All zeros: nothing to replace
        assert np.allclose(sig, cleaned)


# ---
# remove_movement_artifacts
# ---


class TestRemoveMovementArtifacts:
    """Tests for remove_movement_artifacts."""

    def test_shape_preserved_1d(self):
        """Output shape must match input for 1D arrays."""
        sig = _make_sinusoid()
        cleaned = remove_movement_artifacts(sig, sfreq=10.0)
        assert cleaned.shape == sig.shape

    def test_shape_preserved_2d(self):
        """Output shape must match input for 2D arrays."""
        data = np.stack([_make_sinusoid(seed=i) for i in range(4)])
        cleaned = remove_movement_artifacts(data, sfreq=10.0)
        assert cleaned.shape == data.shape

    def test_flat_signal_no_nan(self):
        """A perfectly flat (zero-variance) signal should not produce NaN values.

        The filter always removes the local mean as part of the DC/noise
        estimate, so a constant signal maps to zeros — but crucially must
        not produce NaN from a divide-by-zero.
        """
        sig = np.ones(3000)
        cleaned = remove_movement_artifacts(sig, sfreq=10.0)
        assert not np.any(np.isnan(cleaned))
        # DC component is removed: flat signal → zeros
        assert np.allclose(cleaned, 0.0, atol=1e-10)

    def test_returns_ndarray(self):
        """Output should always be a numpy array."""
        sig = _make_sinusoid()
        cleaned = remove_movement_artifacts(sig, sfreq=10.0)
        assert isinstance(cleaned, np.ndarray)

    def test_gastric_frequency_preserved(self):
        """Gastric-frequency content should be largely preserved after filtering."""
        from scipy.signal import welch

        sfreq = 10.0
        sig = _make_sinusoid(freq=0.05, sfreq=sfreq, duration=300.0)
        cleaned = remove_movement_artifacts(sig, sfreq=sfreq, freq=0.05)

        # Check that the dominant frequency is still near 0.05 Hz
        f, p = welch(cleaned, fs=sfreq, nperseg=min(1024, len(cleaned)))
        gastric_mask = (f >= 0.03) & (f <= 0.07)
        peak_freq = f[gastric_mask][np.argmax(p[gastric_mask])]
        assert np.abs(peak_freq - 0.05) < 0.02

    def test_custom_freq_parameter(self):
        """Custom freq parameter should be accepted without error."""
        sig = _make_sinusoid()
        cleaned = remove_movement_artifacts(sig, sfreq=10.0, freq=0.033)
        assert cleaned.shape == sig.shape
