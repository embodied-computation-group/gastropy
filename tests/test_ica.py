"""Tests for gastropy.signal.ica — ICA-based multichannel denoising."""

import numpy as np
import pytest

from gastropy.signal import ica_denoise


def _make_multichannel(n_channels=4, freq=0.05, sfreq=10.0, duration=300.0, noise_scale=0.2, seed=0):
    """Create multi-channel EGG with shared gastric component and independent noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / sfreq)
    gastric = np.sin(2 * np.pi * freq * t)
    channels = []
    for i in range(n_channels):
        noise = rng.standard_normal(len(t)) * noise_scale
        channels.append(gastric + noise)
    return np.stack(channels)


# ---
# ica_denoise
# ---


class TestIcaDenoise:
    """Tests for ica_denoise."""

    def test_output_shape_unchanged(self):
        """Denoised output must have the same shape as input."""
        data = _make_multichannel(n_channels=4)
        denoised, info = ica_denoise(data, sfreq=10.0)
        assert denoised.shape == data.shape

    def test_info_keys(self):
        """Info dict must contain the expected metadata keys."""
        data = _make_multichannel(n_channels=3)
        _, info = ica_denoise(data, sfreq=10.0)
        for key in ("n_components", "n_kept", "n_removed", "component_snr", "snr_threshold", "band"):
            assert key in info, f"Missing key: {key}"

    def test_n_kept_plus_n_removed_equals_n_components(self):
        """n_kept + n_removed must equal n_components."""
        data = _make_multichannel(n_channels=4)
        _, info = ica_denoise(data, sfreq=10.0)
        assert info["n_kept"] + info["n_removed"] == info["n_components"]

    def test_component_snr_length(self):
        """component_snr array length must equal n_components."""
        data = _make_multichannel(n_channels=4)
        _, info = ica_denoise(data, sfreq=10.0)
        assert len(info["component_snr"]) == info["n_components"]

    def test_at_least_one_component_kept(self):
        """With a clear gastric signal, at least one component should be kept."""
        data = _make_multichannel(n_channels=4, noise_scale=0.05)
        _, info = ica_denoise(data, sfreq=10.0, snr_threshold=2.0)
        assert info["n_kept"] >= 1

    def test_returns_ndarray(self):
        """Denoised output should be a numpy array."""
        data = _make_multichannel(n_channels=3)
        denoised, _ = ica_denoise(data, sfreq=10.0)
        assert isinstance(denoised, np.ndarray)

    def test_raises_on_1d_input(self):
        """Should raise ValueError when given a 1D array."""
        sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 300, 0.1))
        with pytest.raises(ValueError, match="multi-channel"):
            ica_denoise(sig, sfreq=10.0)

    def test_raises_when_all_components_removed(self):
        """Should raise RuntimeError when all components fall below threshold."""
        # Use very high threshold so all components are removed
        data = _make_multichannel(n_channels=3)
        with pytest.raises(RuntimeError, match="All"):
            ica_denoise(data, sfreq=10.0, snr_threshold=1e6)

    def test_random_state_reproducibility(self):
        """Same random_state should produce identical results."""
        data = _make_multichannel(n_channels=4)
        d1, _ = ica_denoise(data, sfreq=10.0, random_state=42)
        d2, _ = ica_denoise(data, sfreq=10.0, random_state=42)
        assert np.allclose(d1, d2)

    def test_explicit_band_frequencies(self):
        """Explicit low_hz/high_hz should override band defaults."""
        data = _make_multichannel(n_channels=3)
        denoised, info = ica_denoise(data, sfreq=10.0, low_hz=0.033, high_hz=0.067)
        assert info["band"]["f_lo"] == pytest.approx(0.033)
        assert info["band"]["f_hi"] == pytest.approx(0.067)

    def test_c_contiguous_output(self):
        """Output array should be C-contiguous (required for downstream Cython-style code)."""
        data = _make_multichannel(n_channels=3)
        denoised, _ = ica_denoise(data, sfreq=10.0)
        assert denoised.flags["C_CONTIGUOUS"]
