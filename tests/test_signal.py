"""Tests for gastropy.signal module."""

import numpy as np
import pytest

from gastropy.signal import (
    apply_bandpass,
    cycle_durations,
    design_fir_bandpass,
    instantaneous_phase,
    mean_phase_per_window,
    psd_welch,
    resample_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=300.0, noise=0.0):
    """Create a sinusoidal test signal."""
    t = np.arange(0, duration, 1.0 / sfreq)
    sig = np.sin(2 * np.pi * freq_hz * t)
    if noise > 0:
        rng = np.random.default_rng(42)
        sig += noise * rng.standard_normal(len(sig))
    return t, sig


# ---------------------------------------------------------------------------
# psd_welch
# ---------------------------------------------------------------------------

class TestPsdWelch:
    def test_peak_at_signal_frequency(self):
        """PSD peak should be at the frequency of the input sinusoid."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=300.0)
        freqs, psd = psd_welch(sig, sfreq=10.0, fmin=0.01, fmax=0.1)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 0.05) < 0.005

    def test_frequency_range_masking(self):
        """Output should only contain frequencies within [fmin, fmax]."""
        _, sig = _make_sinusoid(sfreq=10.0, duration=300.0)
        freqs, psd = psd_welch(sig, sfreq=10.0, fmin=0.02, fmax=0.08)
        assert freqs[0] >= 0.02
        assert freqs[-1] <= 0.08
        assert len(freqs) == len(psd)

    def test_returns_positive_power(self):
        """PSD values should be non-negative."""
        _, sig = _make_sinusoid(sfreq=10.0, duration=300.0)
        _, psd = psd_welch(sig, sfreq=10.0)
        assert np.all(psd >= 0)


# ---------------------------------------------------------------------------
# design_fir_bandpass
# ---------------------------------------------------------------------------

class TestDesignFirBandpass:
    def test_returns_odd_numtaps(self):
        """FIR filter should have an odd number of taps."""
        b, a = design_fir_bandpass(0.03, 0.07, sfreq=10.0)
        assert len(b) % 2 == 1
        np.testing.assert_array_equal(a, [1.0])

    def test_numtaps_capped_at_500(self):
        """Filter length should not exceed 500 taps."""
        b, _ = design_fir_bandpass(0.001, 0.01, sfreq=10.0, f_order=10)
        assert len(b) <= 501  # 500 + possible odd adjustment

    def test_passband_within_nyquist(self):
        """Should handle edge frequencies near Nyquist without error."""
        b, _ = design_fir_bandpass(0.01, 4.9, sfreq=10.0)
        assert len(b) > 0


# ---------------------------------------------------------------------------
# apply_bandpass
# ---------------------------------------------------------------------------

class TestApplyBandpass:
    def test_fir_preserves_in_band_signal(self):
        """FIR filter should preserve a signal within the passband."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=300.0)
        filtered, info = apply_bandpass(sig, sfreq=10.0, low_hz=0.03, high_hz=0.07)
        assert info["filter_method"] == "fir"
        # Correlation with original should be high (signal is in-band)
        corr = np.corrcoef(sig[500:-500], filtered[500:-500])[0, 1]
        assert corr > 0.95

    def test_fir_removes_out_of_band(self):
        """FIR filter should attenuate out-of-band components."""
        t, in_band = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=300.0)
        _, out_band = _make_sinusoid(freq_hz=0.5, sfreq=10.0, duration=300.0)
        mixed = in_band + out_band
        filtered, _ = apply_bandpass(mixed, sfreq=10.0, low_hz=0.03, high_hz=0.07)
        # After filtering, the out-of-band component should be mostly gone
        # Check that filtered matches in_band better than mixed
        corr_filtered = np.corrcoef(in_band[500:-500], filtered[500:-500])[0, 1]
        assert corr_filtered > 0.9

    def test_iir_method(self):
        """IIR filter should also work."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=300.0)
        filtered, info = apply_bandpass(sig, sfreq=10.0, low_hz=0.03, high_hz=0.07, method="iir")
        assert info["filter_method"] == "iir_butter"
        assert len(filtered) == len(sig)

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        _, sig = _make_sinusoid()
        with pytest.raises(ValueError, match="Unknown filter method"):
            apply_bandpass(sig, sfreq=10.0, low_hz=0.03, high_hz=0.07, method="unknown")


# ---------------------------------------------------------------------------
# instantaneous_phase
# ---------------------------------------------------------------------------

class TestInstantaneousPhase:
    def test_phase_is_bounded(self):
        """Phase should be in [-pi, pi]."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=60.0)
        phase, _ = instantaneous_phase(sig)
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

    def test_returns_complex_analytic(self):
        """Analytic signal should be complex."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=60.0)
        _, analytic = instantaneous_phase(sig)
        assert np.iscomplexobj(analytic)
        assert len(analytic) == len(sig)


# ---------------------------------------------------------------------------
# cycle_durations
# ---------------------------------------------------------------------------

class TestCycleDurations:
    def test_known_period(self):
        """Detected cycle durations should match the known period."""
        freq_hz = 0.05  # 20-second period
        sfreq = 10.0
        t, sig = _make_sinusoid(freq_hz=freq_hz, sfreq=sfreq, duration=300.0)
        phase, _ = instantaneous_phase(sig)
        durs = cycle_durations(phase, t)
        # Should detect ~14 cycles in 300s at 20s/cycle
        assert len(durs) > 10
        # Mean duration should be close to 20s
        assert abs(np.mean(durs) - 20.0) < 1.0

    def test_empty_for_short_signal(self):
        """Very short signal should return empty durations."""
        phase = np.array([0.0, 0.5])
        times = np.array([0.0, 0.1])
        durs = cycle_durations(phase, times)
        assert len(durs) == 0


# ---------------------------------------------------------------------------
# mean_phase_per_window
# ---------------------------------------------------------------------------

class TestMeanPhasePerWindow:
    def test_returns_correct_length(self):
        """Should return one phase value per window."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=60.0)
        _, analytic = instantaneous_phase(sig)
        windows = [(0, 100), (100, 200), (200, 300)]
        phases = mean_phase_per_window(analytic, windows)
        assert len(phases) == 3

    def test_out_of_bounds_returns_nan(self):
        """Out-of-bounds windows should return NaN."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=10.0)
        _, analytic = instantaneous_phase(sig)
        windows = [(0, 50), (9999, 10050)]
        phases = mean_phase_per_window(analytic, windows)
        assert not np.isnan(phases[0])
        assert np.isnan(phases[1])


# ---------------------------------------------------------------------------
# resample_signal
# ---------------------------------------------------------------------------

class TestResampleSignal:
    def test_correct_output_length(self):
        """Resampled signal should have approximately the right length."""
        _, sig = _make_sinusoid(sfreq=10.0, duration=100.0)
        resampled, actual_rate = resample_signal(sig, 10.0, 2.0)
        expected_len = int(round(len(sig) * 2.0 / 10.0))
        assert len(resampled) == expected_len

    def test_too_few_samples_raises(self):
        """Should raise ValueError if result would have < 2 samples."""
        sig = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 2"):
            resample_signal(sig, 1000.0, 0.001)

    def test_preserves_frequency_content(self):
        """Resampled signal should preserve the dominant frequency."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=300.0)
        resampled, actual_rate = resample_signal(sig, 10.0, 2.0)
        freqs, psd = psd_welch(resampled, sfreq=actual_rate, fmin=0.01, fmax=0.1)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 0.05) < 0.01
