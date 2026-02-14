"""Tests for gastropy.neuro.fmri module."""

import numpy as np
import pandas as pd
import pytest

from gastropy.neuro.fmri import (
    apply_volume_cuts,
    bold_voxelwise_phases,
    compute_plv_map,
    compute_surrogate_plv_map,
    create_volume_windows,
    find_scanner_triggers,
    phase_per_volume,
    regress_confounds,
)

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


# ---------------------------------------------------------------------------
# Helpers for Layer 2 tests
# ---------------------------------------------------------------------------


def _make_bold_and_confounds(n_voxels=50, n_time=200, seed=42):
    """Create synthetic BOLD data and confound DataFrame."""
    rng = np.random.default_rng(seed)
    bold = rng.standard_normal((n_voxels, n_time))
    confounds = pd.DataFrame(
        {
            "trans_x": rng.standard_normal(n_time),
            "trans_y": rng.standard_normal(n_time),
            "trans_z": rng.standard_normal(n_time),
            "rot_x": rng.standard_normal(n_time),
            "rot_y": rng.standard_normal(n_time),
            "rot_z": rng.standard_normal(n_time),
            "a_comp_cor_00": rng.standard_normal(n_time),
            "a_comp_cor_01": rng.standard_normal(n_time),
            "a_comp_cor_02": rng.standard_normal(n_time),
            "a_comp_cor_03": rng.standard_normal(n_time),
            "a_comp_cor_04": rng.standard_normal(n_time),
            "a_comp_cor_05": rng.standard_normal(n_time),
        }
    )
    # Simulate NaN in first row (typical of fMRIPrep derivatives)
    confounds.iloc[0, 0] = np.nan
    return bold, confounds


# ---------------------------------------------------------------------------
# regress_confounds
# ---------------------------------------------------------------------------


class TestRegressConfounds:
    def test_output_shape(self):
        """Output should match input shape."""
        bold, confounds = _make_bold_and_confounds()
        residuals = regress_confounds(bold, confounds)
        assert residuals.shape == bold.shape

    def test_output_is_zscored(self):
        """Residuals should be approximately z-scored."""
        bold, confounds = _make_bold_and_confounds()
        residuals = regress_confounds(bold, confounds)
        # Mean should be ~0, std should be ~1 for each voxel
        means = residuals.mean(axis=1)
        stds = residuals.std(axis=1)
        np.testing.assert_allclose(means, 0, atol=1e-10)
        np.testing.assert_allclose(stds, 1, atol=0.05)

    def test_custom_confound_cols(self):
        """Should work with a subset of confound columns."""
        bold, confounds = _make_bold_and_confounds()
        residuals = regress_confounds(bold, confounds, confound_cols=["trans_x", "trans_y"])
        assert residuals.shape == bold.shape

    def test_confound_signal_removed(self):
        """BOLD correlated with a confound should have lower correlation after regression."""
        rng = np.random.default_rng(42)
        n_time = 200
        confound_signal = rng.standard_normal(n_time)

        # Create BOLD that's strongly correlated with confound
        bold = np.zeros((1, n_time))
        bold[0] = 5 * confound_signal + rng.standard_normal(n_time)

        confounds = pd.DataFrame(
            {col: rng.standard_normal(n_time) for col in [
                "trans_x", "trans_y", "trans_z",
                "rot_x", "rot_y", "rot_z",
                "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
                "a_comp_cor_03", "a_comp_cor_04", "a_comp_cor_05",
            ]}
        )
        confounds["trans_x"] = confound_signal

        residuals = regress_confounds(bold, confounds)
        corr_before = abs(np.corrcoef(bold[0], confound_signal)[0, 1])
        corr_after = abs(np.corrcoef(residuals[0], confound_signal)[0, 1])
        assert corr_after < corr_before

    def test_mismatched_timepoints_raises(self):
        """Should raise if confounds and BOLD have different timepoints."""
        bold = np.zeros((10, 100))
        confounds = pd.DataFrame({"trans_x": np.zeros(50)})
        with pytest.raises(ValueError, match="timepoints"):
            regress_confounds(bold, confounds, confound_cols=["trans_x"])

    def test_1d_input_raises(self):
        """Should raise for 1D input."""
        with pytest.raises(ValueError, match="2D"):
            regress_confounds(np.zeros(100), pd.DataFrame())


# ---------------------------------------------------------------------------
# bold_voxelwise_phases
# ---------------------------------------------------------------------------


class TestBoldVoxelwisePhases:
    def test_output_shape_no_cuts(self):
        """Output shape should match input without edge cuts."""
        rng = np.random.default_rng(42)
        n_voxels, n_time = 5, 300
        bold = rng.standard_normal((n_voxels, n_time))
        phases = bold_voxelwise_phases(bold, peak_freq_hz=0.05, sfreq=0.5)
        assert phases.shape == (n_voxels, n_time)

    def test_output_shape_with_cuts(self):
        """Output should be trimmed when edge cuts are applied."""
        rng = np.random.default_rng(42)
        n_voxels, n_time = 5, 300
        bold = rng.standard_normal((n_voxels, n_time))
        phases = bold_voxelwise_phases(bold, peak_freq_hz=0.05, sfreq=0.5, begin_cut=20, end_cut=20)
        assert phases.shape == (n_voxels, 260)

    def test_phase_range(self):
        """Phase values should be in [-pi, pi]."""
        rng = np.random.default_rng(42)
        bold = rng.standard_normal((3, 200))
        phases = bold_voxelwise_phases(bold, peak_freq_hz=0.05, sfreq=0.5)
        assert phases.min() >= -np.pi - 1e-10
        assert phases.max() <= np.pi + 1e-10

    def test_sinusoidal_input_produces_valid_phase(self):
        """BOLD with gastric-frequency sinusoid should produce valid phase output."""
        sfreq = 0.5  # 1/TR for TR=2s
        n_time = 400
        t = np.arange(n_time) / sfreq
        freq = 0.05
        bold = np.sin(2 * np.pi * freq * t)[np.newaxis, :]
        phases = bold_voxelwise_phases(bold, peak_freq_hz=freq, sfreq=sfreq)
        assert phases.shape == (1, n_time)
        # Phase should be bounded
        assert phases.min() >= -np.pi - 1e-10
        assert phases.max() <= np.pi + 1e-10
        # Center region should have monotonically increasing unwrapped phase
        center = np.unwrap(phases[0, 100:-100])
        assert np.all(np.diff(center) > 0)

    def test_excessive_cuts_returns_empty(self):
        """Excessive cuts should return empty array."""
        bold = np.zeros((3, 10))
        phases = bold_voxelwise_phases(bold, peak_freq_hz=0.05, sfreq=0.5, begin_cut=5, end_cut=6)
        assert phases.shape == (3, 0)


# ---------------------------------------------------------------------------
# compute_plv_map
# ---------------------------------------------------------------------------


class TestComputePlvMap:
    def test_locked_signals_high_plv(self):
        """Locked EGG and BOLD phases should give high PLV."""
        n_time = 200
        t = np.arange(n_time)
        egg_phase = 2 * np.pi * 0.05 * t
        # All voxels locked to EGG with constant offset
        bold_phases = np.tile(egg_phase, (10, 1)) + 0.3
        plv = compute_plv_map(egg_phase, bold_phases)
        assert plv.shape == (10,)
        np.testing.assert_allclose(plv, 1.0, atol=1e-10)

    def test_random_phases_low_plv(self):
        """Random phases should give low PLV."""
        rng = np.random.default_rng(42)
        egg_phase = rng.uniform(-np.pi, np.pi, 500)
        bold_phases = rng.uniform(-np.pi, np.pi, (10, 500))
        plv = compute_plv_map(egg_phase, bold_phases)
        assert all(plv < 0.15)

    def test_volume_reconstruction(self):
        """Should reconstruct 3D volume with mask."""
        n_time = 100
        egg_phase = np.zeros(n_time)
        bold_phases = np.zeros((5, n_time))

        vol_shape = (3, 3, 3)
        mask = np.zeros(vol_shape, dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 1, 1] = True
        mask[2, 2, 2] = True
        mask[0, 1, 0] = True
        mask[1, 0, 1] = True

        plv_vol = compute_plv_map(egg_phase, bold_phases, vol_shape=vol_shape, mask_indices=mask)
        assert plv_vol.shape == vol_shape
        assert plv_vol[0, 0, 0] == 1.0  # locked phases
        assert plv_vol[0, 0, 1] == 0.0  # outside mask

    def test_mismatched_timepoints_raises(self):
        """Should raise if EGG and BOLD have different timepoints."""
        with pytest.raises(ValueError, match="timepoints"):
            compute_plv_map(np.zeros(100), np.zeros((5, 50)))


# ---------------------------------------------------------------------------
# compute_surrogate_plv_map
# ---------------------------------------------------------------------------


class TestComputeSurrogatePlvMap:
    def test_returns_voxel_values(self):
        """Should return one surrogate PLV per voxel."""
        rng = np.random.default_rng(42)
        n_time = 200
        egg_phase = rng.uniform(-np.pi, np.pi, n_time)
        bold_phases = rng.uniform(-np.pi, np.pi, (5, n_time))
        surr = compute_surrogate_plv_map(egg_phase, bold_phases, n_surrogates=10, seed=42)
        assert surr.shape == (5,)

    def test_surrogate_lower_than_locked_empirical(self):
        """Surrogate should be lower than empirical for locked signals."""
        rng = np.random.default_rng(42)
        n_time = 400
        # Realistic: EGG phase with some jitter (not perfectly periodic)
        base_freq = 0.05
        t = np.arange(n_time)
        # Add frequency modulation so circular shifts actually disrupt coupling
        freq_jitter = base_freq + 0.005 * np.sin(2 * np.pi * 0.003 * t)
        egg_phase = np.cumsum(2 * np.pi * freq_jitter)
        egg_phase = np.angle(np.exp(1j * egg_phase))
        # BOLD locked to EGG with moderate noise
        bold_phases = np.tile(egg_phase, (3, 1)) + 0.5 + 0.3 * rng.standard_normal((3, n_time))

        empirical = compute_plv_map(egg_phase, bold_phases)
        surr = compute_surrogate_plv_map(egg_phase, bold_phases, n_surrogates=50, seed=42)
        assert all(surr < empirical)
