"""Sphinx configuration for the PCAT user documentation."""

project = "PCAT"
author = "Tansu Daylan"
copyright = "2026, Tansu Daylan"
version = "0.1"
release = "0.1.0"

extensions = ["sphinx.ext.mathjax"]
source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "_site", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/pcat_logo.png"
html_favicon = "_static/pcat_logo.png"
htmlhelp_basename = "PCATdoc"

latex_documents = [
    (root_doc, "PCAT.tex", "PCAT Documentation", author, "manual"),
]