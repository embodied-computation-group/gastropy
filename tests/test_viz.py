"""Tests for gastropy.viz — visualization functions."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing

from gastropy.viz import (
    plot_artifacts,
    plot_cycle_histogram,
    plot_egg_comprehensive,
    plot_egg_overview,
    plot_psd,
    plot_tfr,
    plot_volume_phase,
)


@pytest.fixture
def synthetic_signals_df():
    """Create a synthetic signals DataFrame matching egg_process output."""
    n = 3000  # 300s at 10 Hz
    t = np.arange(n) / 10.0
    raw = np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.random.default_rng(42).standard_normal(n)
    filtered = np.sin(2 * np.pi * 0.05 * t)
    phase = np.angle(np.exp(1j * 2 * np.pi * 0.05 * t))
    amplitude = np.abs(np.sin(2 * np.pi * 0.05 * t)) + 0.5
    return pd.DataFrame({"raw": raw, "filtered": filtered, "phase": phase, "amplitude": amplitude})


@pytest.fixture
def synthetic_psd():
    """Create synthetic PSD data."""
    freqs = np.linspace(0, 0.1, 101)
    psd = np.exp(-((freqs - 0.05) ** 2) / (2 * 0.005**2))
    return freqs, psd


@pytest.fixture
def synthetic_artifact_info():
    """Create synthetic artifact info dict."""
    return {
        "artifact_mask": np.zeros(3000, dtype=bool),
        "artifact_segments": [(500, 700), (1500, 1600)],
        "n_artifacts": 2,
    }


class TestPlotPsd:
    """Tests for plot_psd."""

    def test_single_channel(self, synthetic_psd):
        freqs, psd = synthetic_psd
        fig, ax = plot_psd(freqs, psd)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multi_channel(self, synthetic_psd):
        freqs, psd = synthetic_psd
        psd_multi = np.stack([psd, psd * 0.8, psd * 0.5])
        fig, ax = plot_psd(freqs, psd_multi, ch_names=["EGG1", "EGG2", "EGG3"], best_idx=0, peak_freq=0.05)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_existing_ax(self, synthetic_psd):
        import matplotlib.pyplot as plt

        freqs, psd = synthetic_psd
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_psd(freqs, psd, ax=ax_ext)
        assert fig is fig_ext
        assert ax is ax_ext
        plt.close(fig)


class TestPlotEggOverview:
    """Tests for plot_egg_overview."""

    def test_basic(self, synthetic_signals_df):
        fig, axes = plot_egg_overview(synthetic_signals_df, sfreq=10.0)
        assert fig is not None
        assert len(axes) == 4
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_title(self, synthetic_signals_df):
        fig, axes = plot_egg_overview(synthetic_signals_df, sfreq=10.0, title="Test")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotCycleHistogram:
    """Tests for plot_cycle_histogram."""

    def test_basic(self):
        durs = np.array([18.0, 20.0, 22.0, 19.0, 25.0, 16.0, 21.0])
        fig, ax = plot_cycle_histogram(durs)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_durations(self):
        fig, ax = plot_cycle_histogram(np.array([]))
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotArtifacts:
    """Tests for plot_artifacts."""

    def test_basic(self, synthetic_artifact_info):
        phase = np.sin(np.linspace(0, 20 * np.pi, 3000))
        times = np.arange(3000) / 10.0
        fig, ax = plot_artifacts(phase, times, synthetic_artifact_info)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotVolumePhase:
    """Tests for plot_volume_phase."""

    def test_basic(self):
        phase = np.random.default_rng(42).uniform(-np.pi, np.pi, 100)
        fig, ax = plot_volume_phase(phase)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_tr(self):
        phase = np.random.default_rng(42).uniform(-np.pi, np.pi, 100)
        fig, ax = plot_volume_phase(phase, tr=1.856, cut_start=5, cut_end=5)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotEggComprehensive:
    """Tests for plot_egg_comprehensive."""

    def test_basic(self, synthetic_signals_df):
        fig, axes = plot_egg_comprehensive(synthetic_signals_df, sfreq=10.0)
        assert fig is not None
        assert len(axes) == 4
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_volume_phase(self, synthetic_signals_df):
        phase_per_vol = np.random.default_rng(42).uniform(-np.pi, np.pi, 50)
        fig, axes = plot_egg_comprehensive(synthetic_signals_df, sfreq=10.0, phase_per_vol=phase_per_vol, tr=1.856)
        assert fig is not None
        assert len(axes) == 5
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_artifacts(self, synthetic_signals_df, synthetic_artifact_info):
        fig, axes = plot_egg_comprehensive(synthetic_signals_df, sfreq=10.0, artifact_info=synthetic_artifact_info)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotTfr:
    """Tests for plot_tfr."""

    def test_basic(self):
        freqs = np.linspace(0.02, 0.1, 20)
        times = np.arange(0, 300, 0.1)
        power = np.random.default_rng(42).random((len(freqs), len(times)))
        fig, ax = plot_tfr(freqs, times, power)
        assert fig is not None
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_band(self):
        from gastropy.metrics import NORMOGASTRIA

        freqs = np.linspace(0.02, 0.1, 20)
        times = np.arange(0, 100, 0.1)
        power = np.random.default_rng(42).random((len(freqs), len(times)))
        fig, ax = plot_tfr(freqs, times, power, band=NORMOGASTRIA, cmap="hot")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_existing_ax(self):
        import matplotlib.pyplot as plt

        freqs = np.linspace(0.02, 0.1, 10)
        times = np.arange(0, 50, 0.1)
        power = np.random.default_rng(42).random((len(freqs), len(times)))
        fig_ext, ax_ext = plt.subplots()
        fig, ax = plot_tfr(freqs, times, power, ax=ax_ext)
        assert fig is fig_ext
        assert ax is ax_ext
        plt.close(fig)
