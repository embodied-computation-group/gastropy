"""Sphinx configuration for GastroPy documentation."""

import gastropy

# -- Project information ---------------------------------------------------

project = "GastroPy"
copyright = "2026, Micah Allen"
author = "Micah Allen"
version = gastropy.__version__
release = gastropy.__version__

# -- General configuration -------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for autodoc ---------------------------------------------------

autodoc_member_order = "bysource"
autosummary_generate = True
numpydoc_show_class_members = False

# -- Options for myst-nb ---------------------------------------------------

nb_execution_mode = "off"

# -- Options for intersphinx -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mne": ("https://mne.tools/stable/", None),
}

# -- Options for HTML output -----------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/gastropy_logo.png"
html_title = "GastroPy"

html_theme_options = {
    "repository_url": "https://github.com/embodied-computation-group/gastropy",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
}
