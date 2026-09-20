#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NSA-Flow documentation build configuration (following ANTsPy style)

import os
import sys

# Add package root to sys.path
sys.path.insert(0, os.path.abspath('../../'))

import nsa_flow

# Project info
project = 'nsa_flow'
copyright = '2026, B. Avants & NSA-Flow Contributors'
author = 'B. Avants & NSA-Flow Contributors'

version = nsa_flow.__version__
release = nsa_flow.__version__

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
    'sphinx_copybutton',
]

# Generate autosummary stub files
autosummary_generate = True

# Autodoc defaults so module pages list members by default
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Keep object names shorter in headings (omit full module prefix in titles)
add_module_names = False

# Napoleon / docstring parsing
napoleon_use_ivar = True
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Templates & source
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '*.md', 'guides']
suppress_warnings = ['ref.python', 'app.add_directive', 'duplicate_declaration', 'duplicate_object_description']

pygments_style = 'sphinx'

# MyST Parser settings
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'fieldlist',
    'colon_fence',
]

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}
