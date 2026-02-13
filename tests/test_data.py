"""Tests for gastropy.data — sample dataset loaders."""

import numpy as np
import pytest

from gastropy.data import list_datasets, load_egg, load_fmri_egg


class TestLoadFmriEgg:
    """Tests for load_fmri_egg."""

    def test_default_session(self):
        data = load_fmri_egg()
        assert data["signal"].shape[0] == 8
        assert data["signal"].shape[1] > 0
        assert data["sfreq"] == 10.0
        assert len(data["ch_names"]) == 8
        assert data["session"] == "0001"

    def test_trigger_times(self):
        data = load_fmri_egg()
        assert len(data["trigger_times"]) == 420
        assert data["trigger_times"][0] >= 0.0
        assert np.all(np.diff(data["trigger_times"]) > 0)  # monotonically increasing

    def test_tr(self):
        data = load_fmri_egg()
        assert data["tr"] == pytest.approx(1.856)

    @pytest.mark.parametrize("session", ["0001", "0003", "0004"])
    def test_all_sessions(self, session):
        data = load_fmri_egg(session=session)
        assert data["signal"].shape[0] == 8
        assert data["sfreq"] == 10.0
        assert data["session"] == session

    def test_invalid_session(self):
        with pytest.raises(ValueError, match="Unknown session"):
            load_fmri_egg(session="9999")

    def test_signal_dtype(self):
        data = load_fmri_egg()
        assert data["signal"].dtype == np.float64


class TestLoadEgg:
    """Tests for load_egg."""

    def test_shape(self):
        data = load_egg()
        assert data["signal"].shape[0] == 7
        assert data["signal"].shape[1] > 0

    def test_sfreq(self):
        data = load_egg()
        assert data["sfreq"] == 10.0

    def test_channels(self):
        data = load_egg()
        assert len(data["ch_names"]) == 7

    def test_source(self):
        data = load_egg()
        assert data["source"] == "wolpert_2020"

    def test_signal_dtype(self):
        data = load_egg()
        assert data["signal"].dtype == np.float64


class TestListDatasets:
    """Tests for list_datasets."""

    def test_returns_list(self):
        ds = list_datasets()
        assert isinstance(ds, list)
        assert len(ds) == 4

    def test_contains_expected(self):
        ds = list_datasets()
        assert "fmri_egg_session_0001" in ds
        assert "egg_standalone" in ds
