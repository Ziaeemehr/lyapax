from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lyapax  # noqa: E402

project = "lyapax"
copyright = "2026, Abolfazl Ziaeemehr"
author = "Abolfazl Ziaeemehr"
release = lyapax.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # Example files are named NN_topic.py (no plot_ prefix); execute them all.
    "filename_pattern": r"/\d{2}_",
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
    "download_all_examples": False,
}
