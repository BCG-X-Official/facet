# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import itertools
import logging
import re
import sys
from typing import *

import typing_inspect
from sphinx.application import Sphinx
import sphinx

log = logging.getLogger(name=__name__)
log.setLevel(logging.INFO)


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
def _set_paths() -> None:
    import sys
    import os

    module_paths = [
        "facet",
        "pytools",
        "sklearndf",
    ]

    if "cwd" not in globals():
        # noinspection PyGlobalUndefined
        global cwd
        cwd = os.path.join(os.getcwd(), os.pardir, os.pardir)
    print(f"working dir is '{os.getcwd()}'")
    for module_path in module_paths:
        if module_path not in sys.path:
            sys.path.insert(0, os.path.abspath(f"{cwd}/{os.pardir}/{module_path}/src"))
            print(f"added `{sys.path[0]}` to python paths")


_set_paths()

log.info(f"sys.path = {sys.path}")


# fix m2r error
def monkeypatch(cls):
    """ decorator to monkey-patch methods """

    def decorator(f):
        method = f.__name__
        old_method = getattr(cls, method)
        setattr(cls, method, lambda self, *args, **kwargs: f(old_method, self, *args, **kwargs))

    return decorator


# workaround until https://github.com/miyakogi/m2r/pull/55 is merged
@monkeypatch(sphinx.registry.SphinxComponentRegistry)
def add_source_parser(_old_add_source_parser, self, *args, **kwargs):
    # signature is (parser: Type[Parser], **kwargs), but m2r expects
    # the removed (str, parser: Type[Parser], **kwargs).
    if isinstance(args[0], str):
        args = args[1:]
    return _old_add_source_parser(self, *args, **kwargs)



# -- Project information -----------------------------------------------------

project = 'Facet'
copyright = '2020, The Boston Consulting Group (BCG)'
author = 'BCG Gamma Facet Team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "m2r"
]
# -- Options for automodapi ------------------------------------------------------------

# required by default to avoid generating duplicate documentation
numpydoc_show_class_members = False

# document all members from superclasses
automodsumm_inherited_members = True

# draw class inheritance diagrams with GraphViz
automodapi_inheritance_diagram = True

# write generated rst files to disk (for debugging only - no impact on final output)
automodsumm_writereprocessed = False

# -- Options for autodoc / autosummary -------------------------------------------------

# generate autosummary even if no references
autosummary_generate = True

# always overwrite generated autosummaries with newly generated versions
autosummary_generate_overwrite = True

#autodoc_default_options = {
    #"ignore-module-all": True,
    # "inherited-members": True,
    # "show-inheritance": False
#}

nbsphinx_allow_errors = True
nbsphinx_timeout = 60 * 15  # 15 minutes due to tutorial/model notebook

# add intersphinx mapping
intersphinx_mapping = {
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "python": ("https://docs.python.org/3.6", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "shap": ("https://shap.readthedocs.io/en/latest/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# source_parsers = {'.md': CommonMarkParser}

source_suffix = ['.rst', '.md']
# m2r_parse_relative_links = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*/.ipynb_checkpoints/*"]

# -- Options for Math output -----------------------------------------------------------

imgmath_image_format = "svg"
imgmath_use_preview = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = "_static/gamma_logo.jpg"
latex_logo = html_logo

# Class documentation to include docstrings both global to the class, and from __init__
autoclass_content = "both"

# -- End of options section ------------------------------------------------------------


_classes_visited: Set[type] = set()


# noinspection PyUnusedLocal
def add_inheritance(
    app: Sphinx, what: str, name: str, obj: object, options: object, lines: List[str]
) -> None:
    """
    Add list of base classes as the first line of the docstring. Ignore builtin
    classes and
    :param app: the Sphinx application object
    :param what: the type of the object which the docstring belongs to (one of \
        "module", "class", "exception", "function", "method", "attribute")
    :param name: the fully qualified name of the object
    :param obj: the object itself
    :param options: the options given to the directive: an object with attributes \
        inherited_members, undoc_members, show_inheritance and noindex that are true \
        if the flag option of same name was given to the auto directive
    :param lines: the lines of the docstring
    """

    if what == "class" and obj not in _classes_visited:
        # visit each class only once (this method will be called twice if there are
        # docstrings both for the class, and for __init__)

        _classes_visited.add(cast(type, obj))

        # add bases and generics documentation to class

        # generate the RST for bases and generics
        class_ = cast(type, obj)
        bases = list(_get_bases(class_))

        bases_lines = [""]
        if len(bases) > 0:
            base_classes = f':Bases: {", ".join(bases)}'
            bases_lines.append(base_classes)
            bases_lines.append("")

        generics = _get_generics(class_)
        if len(generics) > 0:
            generic_type_variables = f':Generic Types: {", ".join(generics)}'
            bases_lines.append(generic_type_variables)
            bases_lines.append("")

        # insert this after the intro text, and before class parameters

        def _insert_position() -> int:
            for n, line in enumerate(lines):
                if re.match(r"\s*:param\s*\w+:", line):
                    return n
            return -1

        if len(bases_lines) > 0:
            pos = _insert_position()
            lines[pos:pos] = bases_lines


def _class_name(cls: type) -> str:
    try:
        # we try to get the class name
        return cls.__name__
    except AttributeError:
        # if the name is not defined, this class is likely to have generic arguments,
        # so we re-try recursively with the origin (unless the origin is the class
        # itself to avoid infinite recursion)
        cls_origin = typing_inspect.get_origin(cls)
        if cls_origin != cls:
            return _class_name(cls_origin)
        else:
            # as a last resort, we convert the class to a string
            return str(cls)


def _class_name_with_generics(cls: type) -> str:
    if not hasattr(cls, "__module__"):
        return str(cls)
    if cls.__module__ in ("__builtin__", "builtins"):
        return f":class:`{cls.__name__}`"
    else:
        generic_args = [
            _class_name_with_generics(arg)
            for arg in typing_inspect.get_args(cls, evaluate=True)
        ]
        if len(generic_args) == 0:
            generic_arg_str = ""
        else:
            generic_arg_str = f'[{", ".join(generic_args)}]'
        return f":class:`~{cls.__module__}.{_class_name(cls)}` {generic_arg_str}"


def _get_bases(child_class: type) -> Generator[str, None, None]:
    # get the names of the immediate base classes of arg child_class

    # ensure we have the non-generic origin class
    origin = typing_inspect.get_origin(child_class)
    if origin is not None:
        child_class = origin

    # get the base classes, try generic bases first then fall back to "regular" bases
    base_classes: Tuple[type] = typing_inspect.get_generic_bases(child_class)
    if len(base_classes) == 0:
        base_classes = child_class.__bases__

    # get the names of all base classes; go up the class hierarchy in case of hidden
    # classes
    for base in base_classes:
        if base is object or typing_inspect.get_origin(base) is Generic:
            continue
        if _class_name(base).startswith("_"):
            yield from _get_bases(base)
        else:
            yield _class_name_with_generics(base)


def _get_generics(child_class: type) -> List[str]:
    return list(
        itertools.chain.from_iterable(
            (
                [
                    _class_name_with_generics(arg)
                    for arg in typing_inspect.get_args(base, evaluate=True)
                ]
                for base in typing_inspect.get_generic_bases(child_class)
                if typing_inspect.get_origin(base) is Generic
            )
        )
    )


def setup(app: Sphinx) -> None:
    """
    Add event handlers to the Sphinx application object
    :param app: the Sphinx application object
    """
    app.connect("autodoc-process-docstring", add_inheritance)
