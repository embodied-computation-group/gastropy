"""Tests for gastropy.viz.brain — brain map visualization."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

nibabel = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

from gastropy.viz.brain import plot_coupling_map, plot_glass_brain  # noqa: E402


@pytest.fixture
def synthetic_plv_img():
    """Create a minimal synthetic PLV NIfTI image."""
    vol_shape = (10, 12, 8)
    data = np.zeros(vol_shape, dtype=np.float32)
    # Add some non-zero values
    data[3:7, 4:8, 2:6] = 0.05
    data[5, 6, 4] = 0.10
    affine = np.eye(4)
    return nibabel.Nifti1Image(data, affine)


class TestPlotCouplingMap:
    def test_returns_display(self, synthetic_plv_img):
        """Should return a nilearn display object."""
        display = plot_coupling_map(synthetic_plv_img)
        assert display is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_with_threshold(self, synthetic_plv_img):
        """Should accept a threshold parameter."""
        display = plot_coupling_map(synthetic_plv_img, threshold=0.04)
        assert display is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_with_numpy_array(self):
        """Should accept a numpy array + affine."""
        data = np.zeros((10, 12, 8), dtype=np.float32)
        data[5, 6, 4] = 0.1
        affine = np.eye(4)
        display = plot_coupling_map(data, affine=affine)
        assert display is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_custom_cmap(self, synthetic_plv_img):
        """Should accept custom colormap."""
        display = plot_coupling_map(synthetic_plv_img, cmap="hot")
        assert display is not None
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotGlassBrain:
    def test_returns_display(self, synthetic_plv_img):
        """Should return a nilearn display object."""
        display = plot_glass_brain(synthetic_plv_img)
        assert display is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_with_threshold(self, synthetic_plv_img):
        """Should accept a threshold parameter."""
        display = plot_glass_brain(synthetic_plv_img, threshold=0.04)
        assert display is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_with_numpy_array(self):
        """Should accept a numpy array + affine."""
        data = np.zeros((10, 12, 8), dtype=np.float32)
        data[5, 6, 4] = 0.1
        affine = np.eye(4)
        display = plot_glass_brain(data, affine=affine)
        assert display is not None
        import matplotlib.pyplot as plt

        plt.close("all")
