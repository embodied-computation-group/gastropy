"""Tests for gastropy.signal.artifacts — phase-based artifact detection."""

import numpy as np

from gastropy.signal import detect_phase_artifacts, find_cycle_edges, instantaneous_phase
from gastropy.signal.artifacts import _find_duration_outliers, _find_nonmonotonic_cycles


def _make_clean_phase(n_cycles=10, samples_per_cycle=200):
    """Create a clean monotonically-increasing phase with exact wraps."""
    total_samples = n_cycles * samples_per_cycle
    # Sawtooth from -pi to pi, repeating
    phase = np.tile(np.linspace(-np.pi, np.pi, samples_per_cycle, endpoint=False), n_cycles)
    return phase[:total_samples]


class TestFindCycleEdges:
    """Tests for find_cycle_edges."""

    def test_detects_wraps(self):
        phase = _make_clean_phase(n_cycles=5, samples_per_cycle=100)
        edges = find_cycle_edges(phase)
        # Should find 4 edges (between 5 cycles)
        assert len(edges) == 4

    def test_single_cycle(self):
        phase = np.linspace(-np.pi, np.pi, 200)
        edges = find_cycle_edges(phase)
        assert len(edges) == 0

    def test_empty_input(self):
        edges = find_cycle_edges(np.array([]))
        assert len(edges) == 0

    def test_sinusoid_via_hilbert(self):
        sfreq = 10.0
        t = np.arange(0, 300, 1 / sfreq)
        sig = np.sin(2 * np.pi * 0.05 * t)
        phase, _ = instantaneous_phase(sig)
        edges = find_cycle_edges(phase)
        # ~15 cycles expected for 0.05 Hz over 300s
        assert 10 < len(edges) < 20


class TestFindDurationOutliers:
    """Tests for _find_duration_outliers."""

    def test_uniform_durations(self):
        durs = np.array([20.0, 20.0, 20.0, 20.0, 20.0])
        outliers = _find_duration_outliers(durs, sd_threshold=3.0)
        assert len(outliers) == 0

    def test_one_outlier(self):
        # Need enough uniform values so the outlier exceeds mean + 3*SD
        durs = np.array([20.0] * 20 + [100.0])
        outliers = _find_duration_outliers(durs, sd_threshold=3.0)
        assert len(outliers) >= 1
        assert 20 in outliers

    def test_too_few_cycles(self):
        outliers = _find_duration_outliers(np.array([20.0]), sd_threshold=3.0)
        assert len(outliers) == 0


class TestFindNonmonotonicCycles:
    """Tests for _find_nonmonotonic_cycles."""

    def test_clean_phase(self):
        phase = _make_clean_phase(n_cycles=5, samples_per_cycle=100)
        edges = find_cycle_edges(phase)
        bad = _find_nonmonotonic_cycles(phase, edges)
        assert len(bad) == 0

    def test_corrupted_cycle(self):
        phase = _make_clean_phase(n_cycles=5, samples_per_cycle=100)
        edges = find_cycle_edges(phase)
        # Corrupt the 3rd cycle: insert a backward phase jump
        start = edges[1] + 10
        phase[start] = phase[start] - 1.0
        bad = _find_nonmonotonic_cycles(phase, edges)
        assert len(bad) >= 1


class TestDetectPhaseArtifacts:
    """Tests for detect_phase_artifacts."""

    def test_clean_signal_no_artifacts(self):
        phase = _make_clean_phase(n_cycles=10, samples_per_cycle=200)
        times = np.arange(len(phase)) * 0.1
        result = detect_phase_artifacts(phase, times)
        assert result["n_artifacts"] == 0
        assert not np.any(result["artifact_mask"])
        assert len(result["artifact_segments"]) == 0

    def test_returns_expected_keys(self):
        phase = _make_clean_phase(n_cycles=5, samples_per_cycle=200)
        times = np.arange(len(phase)) * 0.1
        result = detect_phase_artifacts(phase, times)
        expected_keys = {
            "artifact_mask",
            "artifact_segments",
            "cycle_edges",
            "cycle_durations_s",
            "n_artifacts",
            "duration_outlier_cycles",
            "nonmonotonic_cycles",
        }
        assert set(result.keys()) == expected_keys

    def test_mask_length_matches_input(self):
        phase = _make_clean_phase(n_cycles=5, samples_per_cycle=200)
        times = np.arange(len(phase)) * 0.1
        result = detect_phase_artifacts(phase, times)
        assert len(result["artifact_mask"]) == len(phase)

    def test_detects_nonmonotonic_artifact(self):
        phase = _make_clean_phase(n_cycles=10, samples_per_cycle=200)
        edges = find_cycle_edges(phase)
        # Corrupt cycle 5: insert backward phase jump
        start = edges[4] + 20
        phase[start] = phase[start] - 2.0
        times = np.arange(len(phase)) * 0.1
        result = detect_phase_artifacts(phase, times)
        assert result["n_artifacts"] >= 1
        assert len(result["nonmonotonic_cycles"]) >= 1

    def test_custom_sd_threshold(self):
        phase = _make_clean_phase(n_cycles=10, samples_per_cycle=200)
        times = np.arange(len(phase)) * 0.1
        # With very tight threshold, some cycles might be flagged
        result_tight = detect_phase_artifacts(phase, times, sd_threshold=0.1)
        result_loose = detect_phase_artifacts(phase, times, sd_threshold=10.0)
        assert result_tight["n_artifacts"] >= result_loose["n_artifacts"]
