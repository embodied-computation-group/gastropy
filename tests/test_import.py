"""Smoke tests for GastroPy package import."""


def test_import():
    """Test that gastropy can be imported."""
    import gastropy

    assert hasattr(gastropy, "__version__")


def test_version():
    """Test that the version string is valid."""
    import gastropy

    assert isinstance(gastropy.__version__, str)
    assert len(gastropy.__version__) > 0
