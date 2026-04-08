# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make sphinx find the package source
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -------------------------------------------------------
project = 'ePDFsuite'
copyright = '2024, Nicolas Ratel-Ramond'
author = 'Nicolas Ratel-Ramond'
release = '0.1.4'

# -- General configuration -----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Generate API docs from docstrings
    'sphinx.ext.autosummary',   # Summary tables for modules/classes
    'sphinx.ext.napoleon',      # Support NumPy and Google docstring styles
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.intersphinx',   # Cross-reference other projects (numpy, scipy…)
    'myst_parser',              # Markdown support (for README etc.)
    'sphinx_autodoc_typehints', # Move type hints from signatures into Parameters
]

# autosummary: auto-generate stub .rst files
autosummary_generate = True

# autodoc defaults
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# napoleon: use NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False   # Keep Parameters section as-is
napoleon_use_rtype = False   # Keep Returns section as-is

# sphinx-autodoc-typehints
always_document_param_types = False   # Don't duplicate if already in docstring
typehints_fully_qualified = False

# intersphinx: link to external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Mock heavy dependencies so Sphinx can import the package without them
autodoc_mock_imports = [
    'hyperspy',
    'hyperspy.api',
    'pyFAI',
    'fabio',
    'pymatgen',
    'pymatgen.core',
    'skimage',
    'skimage.filters',
    'skimage.measure',
    'skimage.morphology',
    'skimage.transform',
    'skimage.feature',
    'ipywidgets',
    'IPython',
    'IPython.display',
    'plotly',
    'h5py',
    'tqdm',
    'streamlit',
]

# myst-parser: allow markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output ---------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'titles_only': False,
}
