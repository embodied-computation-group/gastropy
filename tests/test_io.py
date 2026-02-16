"""Tests for gastropy.io — BIDS physio I/O."""

import gzip
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gastropy.io import parse_bids_filename, read_bids_physio, write_bids_physio

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_signal():
    """Small synthetic signal for round-trip tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 100))


@pytest.fixture()
def bids_pair(tmp_path, sample_signal):
    """Write a BIDS physio file pair and return (tsv_path, json_path, signal)."""
    tsv_path = tmp_path / "sub-01_task-rest_physio.tsv.gz"
    columns = ["EGG1", "EGG2", "EGG3"]
    write_bids_physio(
        tsv_path,
        signal=sample_signal,
        sfreq=10.0,
        columns=columns,
        start_time=0.0,
        Source="test",
    )
    json_path = tmp_path / "sub-01_task-rest_physio.json"
    return tsv_path, json_path, sample_signal


# ---------------------------------------------------------------------------
# TestWriteBidsPhysio
# ---------------------------------------------------------------------------


class TestWriteBidsPhysio:
    def test_creates_files(self, tmp_path, sample_signal):
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        tsv_out, json_out = write_bids_physio(tsv, signal=sample_signal, sfreq=10.0, columns=["a", "b", "c"])
        assert tsv_out.exists()
        assert json_out.exists()

    def test_json_has_required_fields(self, tmp_path, sample_signal):
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        _, json_path = write_bids_physio(tsv, signal=sample_signal, sfreq=10.0, columns=["a", "b", "c"])
        with open(json_path) as f:
            meta = json.load(f)
        assert meta["SamplingFrequency"] == 10.0
        assert meta["StartTime"] == 0.0
        assert meta["Columns"] == ["a", "b", "c"]

    def test_tsv_gz_is_valid_gzip(self, tmp_path, sample_signal):
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        write_bids_physio(tsv, signal=sample_signal, sfreq=10.0, columns=["a", "b", "c"])
        # Should decompress without error
        with gzip.open(tsv, "rt") as f:
            text = f.read()
        lines = text.strip().split("\n")
        assert len(lines) == sample_signal.shape[1]

    def test_column_count_mismatch_raises(self, tmp_path, sample_signal):
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        with pytest.raises(ValueError, match="does not match"):
            write_bids_physio(tsv, signal=sample_signal, sfreq=10.0, columns=["a", "b"])

    def test_extra_json_fields(self, tmp_path, sample_signal):
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        _, json_path = write_bids_physio(
            tsv,
            signal=sample_signal,
            sfreq=10.0,
            columns=["a", "b", "c"],
            TR=1.856,
            Source="test_study",
        )
        with open(json_path) as f:
            meta = json.load(f)
        assert meta["TR"] == 1.856
        assert meta["Source"] == "test_study"

    def test_1d_signal_promoted(self, tmp_path):
        signal_1d = np.ones(50)
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        write_bids_physio(tsv, signal=signal_1d, sfreq=10.0, columns=["ch1"])
        data = read_bids_physio(tsv)
        assert data["signal"].shape == (1, 50)

    def test_creates_parent_dirs(self, tmp_path):
        tsv = tmp_path / "deep" / "nested" / "sub-01_task-rest_physio.tsv.gz"
        signal = np.ones((2, 10))
        write_bids_physio(tsv, signal=signal, sfreq=10.0, columns=["a", "b"])
        assert tsv.exists()


# ---------------------------------------------------------------------------
# TestReadBidsPhysio
# ---------------------------------------------------------------------------


class TestReadBidsPhysio:
    def test_round_trip_float64_precision(self, bids_pair):
        tsv_path, _, original = bids_pair
        data = read_bids_physio(tsv_path)
        np.testing.assert_allclose(data["signal"], original, atol=1e-10)

    def test_inferred_json_path(self, bids_pair):
        tsv_path, _, _ = bids_pair
        data = read_bids_physio(tsv_path)
        assert data["sfreq"] == 10.0
        assert data["columns"] == ["EGG1", "EGG2", "EGG3"]

    def test_explicit_json_path(self, bids_pair):
        tsv_path, json_path, _ = bids_pair
        data = read_bids_physio(tsv_path, json_path=json_path)
        assert data["sfreq"] == 10.0

    def test_extra_json_fields_preserved(self, bids_pair):
        tsv_path, _, _ = bids_pair
        data = read_bids_physio(tsv_path)
        assert data["Source"] == "test"

    def test_missing_tsv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="TSV"):
            read_bids_physio(tmp_path / "nonexistent.tsv.gz")

    def test_missing_json_raises(self, tmp_path):
        # Create TSV but no JSON
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        with gzip.open(tsv, "wt") as f:
            f.write("1.0\t2.0\n")
        with pytest.raises(FileNotFoundError, match="JSON"):
            read_bids_physio(tsv)

    def test_missing_required_field_raises(self, tmp_path):
        tsv = tmp_path / "sub-01_task-rest_physio.tsv.gz"
        json_path = tmp_path / "sub-01_task-rest_physio.json"
        with gzip.open(tsv, "wt") as f:
            f.write("1.0\t2.0\n")
        with open(json_path, "w") as f:
            json.dump({"SamplingFrequency": 10.0}, f)  # missing Columns
        with pytest.raises(ValueError, match="Columns"):
            read_bids_physio(tsv)

    def test_uncompressed_tsv(self, tmp_path, sample_signal):
        # Write as plain .tsv (not .gz)
        tsv = tmp_path / "sub-01_task-rest_physio.tsv"
        json_path = tmp_path / "sub-01_task-rest_physio.json"
        data = sample_signal.T  # (n_samples, n_channels)
        with open(tsv, "w") as f:
            for row in data:
                f.write("\t".join(f"{v:.10g}" for v in row) + "\n")
        with open(json_path, "w") as f:
            json.dump({"SamplingFrequency": 10.0, "Columns": ["a", "b", "c"]}, f)

        result = read_bids_physio(tsv)
        np.testing.assert_allclose(result["signal"], sample_signal, atol=1e-10)


# ---------------------------------------------------------------------------
# TestParseBidsFilename
# ---------------------------------------------------------------------------


class TestParseBidsFilename:
    def test_standard_entities(self):
        result = parse_bids_filename("sub-01_ses-0001_task-rest_physio.tsv.gz")
        assert result["sub"] == "01"
        assert result["ses"] == "0001"
        assert result["task"] == "rest"
        assert result["suffix"] == "physio"
        assert result["extension"] == ".tsv.gz"

    def test_no_session(self):
        result = parse_bids_filename("sub-wolpert_task-rest_physio.tsv.gz")
        assert result["sub"] == "wolpert"
        assert "ses" not in result
        assert result["task"] == "rest"
        assert result["suffix"] == "physio"

    def test_json_extension(self):
        result = parse_bids_filename("sub-01_ses-0001_task-rest_physio.json")
        assert result["extension"] == ".json"
        assert result["suffix"] == "physio"

    def test_entity_without_hyphen_warns(self):
        with pytest.warns(UserWarning, match="no key-value separator"):
            result = parse_bids_filename("sub-01_badpart_task-rest_physio.tsv.gz")
        # The malformed part is skipped, valid entities still parsed
        assert result["sub"] == "01"
        assert result["task"] == "rest"
        assert "badpart" not in result


# ---------------------------------------------------------------------------
# TestBrainvisionToBids
# ---------------------------------------------------------------------------


class TestBrainvisionToBids:
    def test_import_error_without_mne(self):
        with patch.dict("sys.modules", {"mne": None}):
            from gastropy.io._brainvision import brainvision_to_bids

            with pytest.raises(ImportError, match="MNE"):
                brainvision_to_bids("fake.vhdr", "out/", subject="01")

    def test_happy_path_with_mocked_mne(self, tmp_path):
        """Full conversion with a mock MNE Raw object."""
        rng = np.random.default_rng(99)
        fake_signal = rng.standard_normal((2, 50))

        # Build a minimal mock that quacks like mne.io.Raw
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 10.0}
        mock_raw.get_data.return_value = fake_signal
        mock_raw.ch_names = ["EGG1", "EGG2"]

        mock_mne = MagicMock()
        mock_mne.io.read_raw_brainvision.return_value = mock_raw

        with patch.dict("sys.modules", {"mne": mock_mne}):
            from importlib import reload

            import gastropy.io._brainvision as bv_mod

            reload(bv_mod)

            result = bv_mod.brainvision_to_bids(
                "fake.vhdr",
                tmp_path,
                subject="01",
                session="001",
                task="rest",
            )

        # Verify output files exist
        assert result["tsv_path"].exists()
        assert result["json_path"].exists()

        # Verify BIDS directory structure: sub-01/ses-001/beh/
        rel = result["tsv_path"].relative_to(tmp_path)
        assert rel.parts[:3] == ("sub-01", "ses-001", "beh")

        # Verify round-trip data fidelity
        data = read_bids_physio(result["tsv_path"])
        np.testing.assert_allclose(data["signal"], fake_signal, atol=1e-10)
        assert data["columns"] == ["EGG1", "EGG2"]
        assert data["sfreq"] == 10.0

    def test_happy_path_no_session(self, tmp_path):
        """Conversion without session entity omits ses- from path."""
        mock_raw = MagicMock()
        mock_raw.info = {"sfreq": 5.0}
        mock_raw.get_data.return_value = np.ones((1, 20))
        mock_raw.ch_names = ["EGG1"]

        mock_mne = MagicMock()
        mock_mne.io.read_raw_brainvision.return_value = mock_raw

        with patch.dict("sys.modules", {"mne": mock_mne}):
            from importlib import reload

            import gastropy.io._brainvision as bv_mod

            reload(bv_mod)

            result = bv_mod.brainvision_to_bids("fake.vhdr", tmp_path, subject="02")

        rel = result["tsv_path"].relative_to(tmp_path)
        assert rel.parts[:2] == ("sub-02", "beh")
        assert "ses-" not in str(rel)
