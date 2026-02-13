"""Tests for gastropy.neuro.fmri module."""

import numpy as np

from gastropy.neuro.fmri import apply_volume_cuts, create_volume_windows, find_scanner_triggers, phase_per_volume

# ---------------------------------------------------------------------------
# Mock MNE Annotations (no real MNE dependency needed for testing)
# ---------------------------------------------------------------------------


class MockAnnotations:
    """Minimal mock of mne.Annotations for testing."""

    def __init__(self, descriptions, onsets):
        self.description = descriptions
        self.onset = onsets


# ---------------------------------------------------------------------------
# find_scanner_triggers
# ---------------------------------------------------------------------------


class TestFindScannerTriggers:
    def test_exact_label_match(self):
        """Should find triggers matching exact label."""
        annot = MockAnnotations(
            descriptions=["R128", "R128", "other", "R128"],
            onsets=[0.0, 1.856, 3.0, 3.712],
        )
        onsets = find_scanner_triggers(annot, label="R128")
        assert len(onsets) == 3
        np.testing.assert_allclose(onsets, [0.0, 1.856, 3.712])

    def test_suffix_match(self):
        """Should match '/R128' suffix patterns."""
        annot = MockAnnotations(
            descriptions=["path/to/R128", "other"],
            onsets=[5.0, 10.0],
        )
        onsets = find_scanner_triggers(annot, label="R128")
        assert len(onsets) == 1
        assert onsets[0] == 5.0

    def test_no_matches(self):
        """Should return empty array if no triggers found."""
        annot = MockAnnotations(
            descriptions=["stimulus", "response"],
            onsets=[1.0, 2.0],
        )
        onsets = find_scanner_triggers(annot, label="R128")
        assert len(onsets) == 0

    def test_sorted_output(self):
        """Output should be sorted by onset time."""
        annot = MockAnnotations(
            descriptions=["R128", "R128", "R128"],
            onsets=[5.0, 1.0, 3.0],
        )
        onsets = find_scanner_triggers(annot, label="R128")
        np.testing.assert_array_equal(onsets, [1.0, 3.0, 5.0])


# ---------------------------------------------------------------------------
# create_volume_windows
# ---------------------------------------------------------------------------


class TestCreateVolumeWindows:
    def test_correct_number_of_windows(self):
        """Should create the requested number of windows."""
        tr = 1.856
        onsets = np.arange(0, 100, tr)
        windows = create_volume_windows(onsets, tr=tr, n_volumes=10)
        assert len(windows) == 10

    def test_capped_at_n_volumes(self):
        """Should not exceed n_volumes even if more onsets available."""
        onsets = np.arange(0, 100, 1.0)
        windows = create_volume_windows(onsets, tr=1.0, n_volumes=5)
        assert len(windows) == 5

    def test_window_indices_are_tuples(self):
        """Each window should be a (start, end) tuple."""
        onsets = np.array([0.0, 2.0, 4.0])
        windows = create_volume_windows(onsets, tr=2.0, n_volumes=3)
        for start, end in windows:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert end > start


# ---------------------------------------------------------------------------
# phase_per_volume
# ---------------------------------------------------------------------------


class TestPhasePerVolume:
    def test_returns_phase_per_window(self):
        """Should return one phase per window."""
        from gastropy.signal import instantaneous_phase

        sig = np.sin(2 * np.pi * 0.05 * np.arange(0, 100, 0.1))
        _, analytic = instantaneous_phase(sig)
        windows = [(0, 100), (100, 200), (200, 300)]
        phases = phase_per_volume(analytic, windows)
        assert len(phases) == 3
        assert all(not np.isnan(p) for p in phases)


# ---------------------------------------------------------------------------
# apply_volume_cuts
# ---------------------------------------------------------------------------


class TestApplyVolumeCuts:
    def test_correct_trimming(self):
        """Should remove begin_cut from start and end_cut from end."""
        data = np.arange(420)
        trimmed = apply_volume_cuts(data, begin_cut=21, end_cut=21)
        assert len(trimmed) == 378
        assert trimmed[0] == 21
        assert trimmed[-1] == 398

    def test_zero_end_cut(self):
        """End cut of 0 should only trim the beginning."""
        data = np.arange(100)
        trimmed = apply_volume_cuts(data, begin_cut=10, end_cut=0)
        assert len(trimmed) == 90
        assert trimmed[0] == 10

    def test_excessive_cuts_returns_empty(self):
        """Cuts exceeding data length should return empty array."""
        data = np.arange(10)
        trimmed = apply_volume_cuts(data, begin_cut=5, end_cut=6)
        assert len(trimmed) == 0
