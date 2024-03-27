# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import dpctl

sys.path.insert(0, os.path.abspath("."))

import extlinks_gen as urlgen  # noqa: E402

project = "Data Parallel Control (dpctl)"
copyright = "2020-2024, Intel Corp."
author = "Intel Corp."

version = dpctl.__version__.strip(".dirty")
# The full version, including alpha/beta/rc tags
release = dpctl.__version__.strip(".dirty")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput",
    # "sphinxcontrib.googleanalytics",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

highlight_language = "Python"

source_suffix = ".rst"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    # "canonical_url": "",
    # "analytics_id": "",
    # "display_version": True,
    # "prev_next_buttons_location": "bottom",
    # "style_external_links": False,
    # "logo_only": False,
    # Toc options
    # "collapse_navigation": True,
    # "sticky_navigation": True,
    # "navigation_depth": 4,
    # "includehidden": True,
    # "titles_only": False,
}


# A dictionary of urls
extlinks = urlgen.create_extlinks()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "numba_dpex": ("https://intelpython.github.io/numba-dpex/latest/", None),
    "cython": ("https://docs.cython.org/en/latest/", None),
}
