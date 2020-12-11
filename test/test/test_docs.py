import importlib
import inspect
import logging
import os
import re
from glob import glob
from typing import List, Tuple

# noinspection PyPackageRequirements
import pytest

log = logging.getLogger(__name__)


@pytest.fixture
def all_script_files() -> List[str]:
    """
    Fixture returning all .py script files in the src/yieldengine codebase

    :return: List of relative paths
    """
    start_dir = os.path.join(os.pardir, "yieldengine")
    return [y for x in os.walk(start_dir) for y in glob(os.path.join(x[0], "*.py"))]


@pytest.fixture
def all_script_objects(all_script_files) -> List[object]:
    """
    Dynamically import all Python modules in the codebase

    :param all_script_files: List of all relevant script paths
    :return: List of imported Python modules as objects
    """
    modules = [
        s[3:-3].replace("/", ".").replace(".__init__", "") for s in all_script_files
    ]
    return [(m, importlib.import_module(m)) for m in modules]


# add function, method, or class names in here to ignore from being checked
IGNORE_LIST_CLASS = []
IGNORE_LIST_FUNCTION = [
    "__repr__",
    "__getitem__",
    "__len__",
    "__setitem__",
    "__call__",
    "__new__",
    "__getstate__",
    "__setstate__",
    "set_params",
]
IGNORE_LIST_METHOD = []


def member_childs(
    member: object, module_filter: None
) -> Tuple[
    List[Tuple[str, object]], List[Tuple[str, object]], List[Tuple[str, object]]
]:
    """
    Return childs of a given member (module, class,..)
    :param member: a Python module or class
    :param module_filter: an optional filter for __module__ of the childs
    :return: three lists for classes, functions and methods
    """

    all_childs = inspect.getmembers(member)

    if module_filter is not None:
        all_childs = [
            c
            for c in all_childs
            if hasattr(c[1], "__module__") and c[1].__module__ == module_filter
        ]

    classes = [
        c for c in all_childs if inspect.isclass(c[1]) and c[0] not in IGNORE_LIST_CLASS
    ]
    functions = [
        c
        for c in all_childs
        if inspect.isfunction(c[1]) and c[0] not in IGNORE_LIST_FUNCTION
    ]
    methods = [
        c
        for c in all_childs
        if inspect.ismethod(c[1]) and c[0] not in IGNORE_LIST_METHOD
    ]

    return classes, functions, methods


def docstr_missing(obj_name: str, obj: object) -> bool:
    """
    Check if __doc__ is missing or empty
    :param obj_name: name of the object to check
    :param obj: object to check
    :return: boolean if docstr is missing
    """
    if hasattr(obj, "__doc__") and not obj_name.startswith("__"):
        if obj.__doc__ is None or len(str(obj.__doc__).strip()) < 1:
            return True

    return False


def extract_params_from_docstr(docstr: str) -> List[str]:
    """
    Extract all documented parameter names from a docstring

    :param docstr: the input docstring
    :return: list of parameter names
    """
    all_params = re.findall(
        pattern="(\\:param\\s+)(.{1,40})(\\:)", string=docstr, flags=re.MULTILINE
    )

    return [p[1].strip() for p in all_params]


def parameters_inconsistent(parent: str, call_obj_name: str, call_obj: object) -> bool:
    """
    Check if parameters are inconsistent between a callable's signature and docstr

    :param parent: Name of the module/class the callable appears in (for log)
    :param call_obj_name: the name of the callable to check
    :param call_obj: the callable to check
    :return: True if inconsistent, else False
    """
    docstr_params = extract_params_from_docstr(str(call_obj.__doc__))
    full_args = inspect.getfullargspec(call_obj)
    func_args = full_args.args

    if "self" in func_args:
        func_args.remove("self")

    if docstr_params is not None and len(docstr_params) > 0:

        for idx, f_arg in enumerate(func_args):
            if idx > len(docstr_params) or docstr_params[idx] != f_arg:
                log.info(
                    f"Wrong arguments in docstr for {parent}.{call_obj_name}: {f_arg}"
                )
                return True
    else:
        # the function has arguments defined but none were found from __doc__?
        if len(full_args.args) > 0:
            log.info(f"No documented arguments in docstr for {parent}.{call_obj_name}")
            return True

    # all ok
    return False


def test_docstrings(all_script_objects) -> None:

    classes_with_missing_docstr = []
    functions_with_missing_docstr = []
    methods_with_missing_docstr = []

    inconsistent_parameters = []

    for module_name, module in all_script_objects:

        classes, functions, methods = member_childs(module, module_filter=module_name)

        # classes where docstring is None:
        classes_with_missing_docstr.extend(
            [name for name, obj in classes if docstr_missing(name, obj)]
        )

        # functions where docstring is None:
        functions_with_missing_docstr.extend(
            [name for name, obj in functions if docstr_missing(name, obj)]
        )

        # methods where docstring is None:
        methods_with_missing_docstr.extend(
            [name for name, obj in methods if docstr_missing(name, obj)]
        )

        for f_name, f_obj in functions:
            if not docstr_missing(f_name, f_obj) and parameters_inconsistent(
                module_name, f_name, f_obj
            ):
                inconsistent_parameters.append(f_name)

        # inspect found classes:
        for clsname, cls in classes:
            _inner_classes, inner_functions, inner_methods = member_childs(cls, None)

            # functions where docstring is None:
            functions_with_missing_docstr.extend(
                [name for name, obj in inner_functions if docstr_missing(name, obj)]
            )

            for f_name, f_obj in inner_functions:
                if not docstr_missing(f_name, f_obj) and parameters_inconsistent(
                    clsname, f_name, f_obj
                ):
                    inconsistent_parameters.append(clsname + "." + f_name)

            # methods where docstring is None:
            methods_with_missing_docstr.extend(
                [name for name, obj in inner_methods if docstr_missing(name, obj)]
            )

    if classes_with_missing_docstr:
        log.info(f"The following classes lack-docstrings:{classes_with_missing_docstr}")

    if functions_with_missing_docstr:
        log.info(
            f"The following functions lack-docstrings:{functions_with_missing_docstr}"
        )

    if methods_with_missing_docstr:
        log.info(f"The following methods lack-docstrings:{methods_with_missing_docstr}")

    if inconsistent_parameters:
        log.info(
            f"The following methods have inconsistently described parameters:"
            f"{inconsistent_parameters}"
        )

    assert len(classes_with_missing_docstr) == 0
    assert len(functions_with_missing_docstr) == 0
    assert len(methods_with_missing_docstr) == 0
    assert len(inconsistent_parameters) == 0
