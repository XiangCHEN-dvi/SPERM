from datetime import UTC, datetime
from importlib.metadata import version as package_version

project = "SPERM"
author = "Xiang CHEN"
copyright = f"{datetime.now(UTC).year}, {author}"
release = package_version("sperm")
version = release

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
nitpicky = True
nitpick_ignore_regex = [
    ("py:class", r"default=.*"),
    ("py:class", r"mapping"),
]

html_theme = "pydata_sphinx_theme"
html_title = "SPERM"
html_baseurl = "https://xiangchen-dvi.github.io/sperm/"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [("custom-icons.js", {"defer": "defer"})]
html_theme_options = {
    "github_url": "https://github.com/XiangCHEN-dvi/sperm",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/sperm/",
            "icon": "fa-custom fa-pypi",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
    "navigation_with_keys": True,
    "show_prev_next": False,
    "show_toc_level": 2,
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": ["build-info"],
}
html_context = {
    "github_user": "XiangCHEN-dvi",
    "github_repo": "sperm",
    "github_version": "main",
    "doc_path": "docs",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
