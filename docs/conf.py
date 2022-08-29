# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from sphinx.ext.autodoc import ClassDocumenter, _

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Tyche'
copyright = '2022, Padraig Lamont, drtnf'
author = 'Padraig Lamont, drtnf'
release = 'v1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'autoapi.extension', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []

sys.path.insert(0, os.path.abspath('..'))
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike'
}
autoclass_content = 'both'

autoapi_type = 'python'
autoapi_options = ['show-inheritance']
autoapi_member_order = 'bysource'
autoapi_dirs = ['../tyche']
autoapi_template_dir = '_templates'
autoapi_generate_api_docs = False
autoapi_root = "tyche"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_logo = 'logo.png'
html_favicon = 'favicon.ico'

# ClassDocumenter.add_directive_header uses ClassDocumenter.add_line to
# write the class documentation.
# We'll monkeypatch the add_line method and intercept lines that begin
# with "Bases:".
# In order to minimize the risk of accidentally intercepting a wrong line,
# we'll apply this patch inside of the add_directive_header method.

add_line = ClassDocumenter.add_line
lines_to_delete = [
    _(u'Bases: %s') % u':py:class:`object`',
    _(u'Bases: %s') % u':py:class:`Exception`'
]


def add_line_no_object_base(self, text, *args, **kwargs):
    if text.strip() in lines_to_delete:
        return

    add_line(self, text, *args, **kwargs)


add_directive_header = ClassDocumenter.add_directive_header


def add_directive_header_no_object_base(self, *args, **kwargs):
    self.add_line = add_line_no_object_base.__get__(self)

    result = add_directive_header(self, *args, **kwargs)

    del self.add_line

    return result


ClassDocumenter.add_directive_header = add_directive_header_no_object_base
