# Contributing to GastroPy

Thank you for your interest in contributing to GastroPy! Whether you're reporting a bug, suggesting a feature, improving documentation, or writing code, your help is appreciated.

## Ways to Contribute

- **Bug reports** — open an [issue](https://github.com/embodied-computation-group/gastropy/issues) with a minimal reproducible example
- **Feature requests** — open an issue describing the use case and expected behavior
- **Documentation** — fix typos, improve docstrings, add examples or tutorials
- **Code** — fix bugs, implement new features, improve test coverage

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/gastropy.git
cd gastropy
```

2. Create a virtual environment and install in development mode:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

To work on neuroimaging features (`gastropy.neuro`) or documentation, install the full dependency set:

```bash
pip install -e ".[all]"
```

3. Install pre-commit hooks (auto-formats code on each commit):

```bash
pre-commit install
```

4. Verify your setup:

```bash
pytest
```

All tests should pass.

## Workflow

1. Create a feature branch off `main`:

```bash
git checkout -b my-feature main
```

2. Make your changes in small, focused commits.

3. Push and open a pull request against `main`:

```bash
git push origin my-feature
```

Keep PRs focused — one logical change per PR is easier to review than a large omnibus.

## Code Style

GastroPy uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

**Key settings** (configured in `pyproject.toml`):

- Line length: 120 characters
- Target Python: 3.10+
- Enabled rule sets: pycodestyle, pyflakes, isort, pyupgrade, flake8-bugbear, flake8-simplify, NumPy

Pre-commit hooks will auto-fix most style issues when you commit. You can also run them manually:

```bash
ruff check --fix gastropy/
ruff format gastropy/
```

## Testing

All new public functions need tests. Run the full suite with:

```bash
pytest
```

**Conventions:**

- Test files live in `tests/` and are named `test_<module>.py`.
- Group tests by function using classes: `class TestFunctionName`.
- Prefix test helper functions with `_` (e.g., `def _make_sinusoid(...)`).
- Write a short docstring on each test method explaining expected behavior.
- Separate sections with comment banners:

```python
# ---------------------------------------------------------------------------
# function_name
# ---------------------------------------------------------------------------
```

**Example:**

```python
class TestPsdWelch:
    def test_peak_at_signal_frequency(self):
        """PSD peak should be at the frequency of the input sinusoid."""
        _, sig = _make_sinusoid(freq_hz=0.05, sfreq=10.0, duration=300.0)
        freqs, psd = psd_welch(sig, sfreq=10.0, fmin=0.01, fmax=0.1)
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 0.05) < 0.005
```

Run a single test file or test class:

```bash
pytest tests/test_signal.py
pytest tests/test_signal.py::TestPsdWelch
```

## Documentation

### Docstrings

Use **NumPy-style** docstrings with `Parameters`, `Returns`, and `Examples` sections:

```python
def psd_welch(data, sfreq, fmin=0.0, fmax=0.1, overlap=0.25):
    """Compute power spectral density using Welch's method.

    Parameters
    ----------
    data : array_like
        Input signal (1D array).
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz.
    psd : np.ndarray
        Power spectral density values.

    Examples
    --------
    >>> freqs, psd = psd_welch(sig, sfreq=10.0)
    """
```

### Building docs locally

```bash
pip install -e ".[docs]"
sphinx-build docs docs/_build -b html
```

Then open `docs/_build/index.html` in your browser.

### Tutorials and examples

- **Tutorials** (`docs/tutorials/`) — longer narrative notebooks walking through a complete analysis
- **Examples** (`docs/examples/`) — short, focused notebooks demonstrating individual functions

Notebooks should be pre-executed (the Sphinx build runs with `nb_execution_mode = "off"`).

## Architecture Guidelines

Understanding these principles will help your contributions fit naturally into the codebase.

### Core modules are dependency-light

The core modules — `signal`, `metrics`, `egg`, `coupling`, `timefreq` — depend only on numpy, scipy, pandas, and matplotlib. They must **not** import MNE, nilearn, nibabel, or other neuroimaging libraries. scikit-learn is an **optional** dependency available via `pip install gastropy[ica]`; it is imported lazily inside `ica_denoise` and raises a clear `ImportError` with install instructions if absent.

### Neuroimaging features are isolated

Anything that requires MNE, nilearn, or nibabel belongs in `gastropy.neuro.*` subpackages (e.g., `gastropy.neuro.fmri`). These dependencies are optional and installed via `pip install gastropy[neuro]`.

### Flat namespace

Public functions are re-exported from `gastropy/__init__.py` so users can write `from gastropy import psd_welch` instead of `from gastropy.signal.spectral import psd_welch`. When adding a new public function, add it to the relevant module's `__all__` list.

### Composable over monolithic

Prefer small, focused functions that take numpy arrays and return numpy arrays. Users should be able to combine them freely rather than being locked into a single pipeline.

## CI Checks

Every pull request must pass:

| Check | What it does |
|-------|-------------|
| **Tests** | `pytest` on Python 3.10, 3.11, 3.12, 3.13 (Ubuntu) |
| **Lint** | `ruff check gastropy/` and `ruff format --check gastropy/` |
| **Docs** | `sphinx-build` (runs on push to `main`) |

## Commit Messages

- Use imperative mood: "Add feature", "Fix bug", not "Added" or "Fixes"
- Keep the headline concise (under ~72 characters)
- Add a body for non-trivial changes explaining *why*, not just *what*

```
Add instability coefficient to metrics module

Implements the coefficient of variation of cycle durations,
following the method described in Koch & Stern (2004).
```

## Releases

Releases are handled by the maintainers. If you think a release is needed, open an issue.

## Questions?

If anything is unclear, open an [issue](https://github.com/embodied-computation-group/gastropy/issues) or start a [discussion](https://github.com/embodied-computation-group/gastropy/discussions). We're happy to help.
