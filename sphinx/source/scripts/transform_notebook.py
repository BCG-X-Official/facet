#!/usr/bin/env python3

"""
Several helpers to deal with Jupyter notebook conversions
"""
import glob
import json
import os
import sys
from typing import Any, Dict

# Jupyter's field keys:

K_SOURCE = "source"
K_CELLS = "cells"
K_CELL_TYPE = "cell_type"
K_METADATA = "metadata"


def docs_notebook_to_interactive(path_in: str, path_out: str = None):
    """ Convert a Sphinx docs notebook to an interactive one. """
    store_notebook(
        data=raw_cells_to_markdown(
            notebook=replace_sphinx_keywords(
                notebook=delete_cells_for_interactive(notebook=read_notebook(path_in))
            )
        ),
        path_out=path_out,
    )


def delete_cells_for_interactive(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """ Delete cells we have marked in their metadata. """
    notebook[K_CELLS] = [
        c
        for c in notebook[K_CELLS]
        if (not K_METADATA in c)
        or not c[K_METADATA].get("delete_for_interactive", False)
    ]
    return notebook


def replace_sphinx_keywords(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """ Replaces several sphinx autodoc keywords, i.e. ":class:", ":mod:" """
    new_cells = []
    for c in notebook[K_CELLS]:
        c[K_SOURCE] = [
            s.replace(":class:", "")
            .replace(":mod:", "")
            .replace(":attr:", "")
            .replace(":meth:", "")
            .replace(":module:", "")
            .replace(":func:", "")
            for s in c[K_SOURCE]
        ]
        new_cells.append(c)

    notebook[K_CELLS] = new_cells

    return notebook


def raw_cells_to_markdown(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """ Changes raw Notebook cells to markdown cells for proper rendering """
    new_cells = []
    for c in notebook[K_CELLS]:
        if c[K_CELL_TYPE] == "raw":
            c[K_CELL_TYPE] = "markdown"

        new_cells.append(c)

    notebook[K_CELLS] = new_cells

    return notebook


def read_notebook(path_in: str) -> Dict[str, Any]:
    """ Reads an .ipynb notebook to a Dict. """
    with open(path_in, "rb") as file:
        return json.load(fp=file)


def store_notebook(data: Dict[str, Any], path_out: str) -> None:
    """ Stores an .ipynb notebook. """
    with open(path_out, "w") as file:
        json.dump(data, file, indent=4)


def docs_notebooks_to_interactive(path_in: str, path_out: str = None) -> None:
    """ Transforms multiple docs notebooks to interactive notebooks """
    if not os.path.isdir(path_in):
        raise ValueError(f"Input path {path_in} should be a dir.")
    if not os.path.isdir(path_out):
        raise ValueError(f"Output path {path_out} should be a dir.")

    for nb in glob.glob(os.path.join(path_in, "*.ipynb")):
        f_path_in = os.path.abspath(nb)
        f_path_out = os.path.abspath(
            os.path.join(path_out, os.path.basename(f_path_in))
        )
        print(f"Converting {f_path_in} and saving to {f_path_out}.")
        docs_notebook_to_interactive(path_in=f_path_in, path_out=f_path_out)


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage: input-folder output-folder")
        sys.exit(2)  # 2 -> command line syntax error

    docs_notebooks_to_interactive(sys.argv[1], sys.argv[2])
