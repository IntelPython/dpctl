#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2021 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

""" The module provides helper functions to generate API documentation for
    dpctl's Python classes.
"""

import enum
import inspect
import io
import os
import sys
from importlib import import_module
from pkgutil import iter_modules

import dpctl

# known property in Cython extension class
_getset_descriptor = type(dpctl.SyclDevice.name)
# known method (defined using def in Cython extension class)
_cython_method_type = type(dpctl.SyclDevice.get_filter_string)
# known builtin method (defined using cpdef in Cython extension class)
_cython_builtin_function_or_method_type = type(dpctl.SyclQueue.mro)

# Dictionary mapping internal module names to a readable string. so that we
# can use the module name to logically group functions.
function_groups = {
    "dpctl._sycl_device_factory": "Device Selection Functions",
    "dpctl._device_selection": "Device Selection Functions",
    "dpctl._sycl_queue_manager": "Queue Management Functions",
}


def _get_module(module):

    try:
        return sys.modules[module]
    except KeyError:
        raise ValueError(
            module + "is not a valid module name or it is not loaded"
        )


def _write_line(output, s):
    output.write(s)
    output.write("\n")


def _write_empty_line(output):
    _write_line(output, "")


def _write_marquee(o, s):
    marquee = "#" * len(s)
    _write_line(o, marquee)
    _write_line(o, s)
    _write_line(o, marquee)


def _write_underlined(o, s, c):
    _write_line(o, s)
    _write_line(o, c * len(s))


def _get_public_class_name(cls):
    if not inspect.isclass(cls):
        raise TypeError("Expecting class, got {}".format(type(cls)))
    modl = cls.__module__
    if modl:
        modl = ".".join(
            [comp for comp in modl.split(".") if not comp.startswith("_")]
        )
    if modl:
        res = ".".join([modl, cls.__qualname__])
    else:
        res = cls.__qualname__
    return res


def _is_class_property(o):
    return isinstance(o, property) or (type(o) == _getset_descriptor)


def _is_class_method(o):
    return inspect.ismethod(o) or (
        type(o)
        in [_cython_method_type, _cython_builtin_function_or_method_type]
    )


def _get_filtered_names(cls, selector_func):
    return [
        _name
        for _name, _obj in inspect.getmembers(cls, selector_func)
        if not _name.startswith("__")
    ]


def _group_functions(mod):
    """Bin module functions into a set of logical groups.

    Args:
        mod (object): A module whose functions will be grouped into bins
            based on the ``function_groups`` dictionary.

    Returns:
        [dict]: A dictionary containing  grouping of functions in the
                module.
    """
    groups = {}
    for name, obj in inspect.getmembers(mod):
        if inspect.isbuiltin(obj) or inspect.isfunction(obj):
            if obj.__module__ and obj.__module__ in function_groups:
                try:
                    flist = groups[function_groups[obj.__module__]]
                    flist.append(obj)
                except KeyError:
                    groups[function_groups[obj.__module__]] = [
                        obj,
                    ]
            else:
                try:
                    flist = groups["Other Functions"]
                    flist.append(obj)
                except KeyError:
                    groups["Other Functions"] = [
                        obj,
                    ]
    return groups


def generate_class_rst(cls):
    """Generate a rst file with the API documentation for a class.

    Raises:
        TypeError: When the input is not a Python class

    Returns:
        [str]: A string with rst nodes that can be written out to a file.
    """
    if not inspect.isclass(cls):
        raise TypeError("Expecting class, got {}".format(type(cls)))

    cls_qualname = _get_public_class_name(cls)
    rst_header = cls_qualname.split(".")[-1]
    rst_module = ".".join(cls_qualname.split(".")[:-1])
    rst_header = "".join([".. _", rst_header, "_api:"])

    def write_rubric(o, indent, rubric_display, rubric_tag, cls_qualname):
        _write_line(o, indent + ".. rubric:: " + rubric_display)
        _write_empty_line(o)
        _write_line(o, indent + ".. autoautosummary:: " + cls_qualname)
        _write_line(o, indent + indent + ":" + rubric_tag + ":")
        _write_empty_line(o)

    with io.StringIO() as output:
        _write_line(output, rst_header)
        _write_empty_line(output)
        _write_marquee(output, cls_qualname)
        _write_empty_line(output)

        _write_line(output, ".. currentmodule:: " + rst_module)
        _write_empty_line(output)

        _write_line(output, ".. autoclass:: " + cls_qualname)
        _write_empty_line(output)

        indent = "    "

        attributes_header = "Attributes"
        write_rubric(
            output, indent, attributes_header + ":", "attributes", cls_qualname
        )
        public_methods_header = "Public methods"
        write_rubric(
            output, indent, public_methods_header + ":", "methods", cls_qualname
        )
        private_methods_header = "Private methods"
        write_rubric(
            output,
            indent,
            private_methods_header + ":",
            "private_methods",
            cls_qualname,
        )

        _write_underlined(output, "Detail", "=")
        _write_empty_line(output)

        # Attributes
        all_attributes = _get_filtered_names(cls, _is_class_property)
        if all_attributes:
            _write_underlined(output, attributes_header, "-")
            _write_empty_line(output)
            for n in all_attributes:
                _write_line(
                    output,
                    ".. autoattribute:: " + ".".join([cls_qualname, n]),
                )
            _write_empty_line(output)

        # Methods, separated into public/private
        all_methods = _get_filtered_names(cls, _is_class_method)
        all_public_methods = []
        all_private_methods = []
        for _name in all_methods:
            if _name.startswith("_"):
                all_private_methods.append(_name)
            else:
                all_public_methods.append(_name)

        if all_public_methods:
            _write_underlined(output, public_methods_header, "-")
            _write_empty_line(output)
            for n in all_public_methods:
                _write_line(
                    output,
                    ".. autoattribute:: " + ".".join([cls_qualname, n]),
                )
            _write_empty_line(output)

        # Private methods
        if all_private_methods:
            _write_underlined(output, private_methods_header, "-")
            _write_empty_line(output)
            for n in all_private_methods:
                _write_line(
                    output,
                    ".. autoattribute:: " + ".".join([cls_qualname, n]),
                )

        return output.getvalue()


def generate_module_summary_rst(module):
    """[summary]

    Args:
        module ([str]): [description]

    Returns:
        [type]: [description]
    """
    rst_header = "".join([".. _", module, "_pyapi:"])
    pagename = module + " API"
    indent = "    "

    def _safe_get_docs(obj, i=0):
        docstr = getattr(obj, "__doc__")
        if not isinstance(docstr, str):
            docstr = f"[FIXME]: {type(obj)} does not have a docstring"
            return docstr
        docstr = docstr.split("\n")
        if len(docstr) < i + 1:
            return f"[FIXME]: {type(obj)} has a docstring with no summary"
        return docstr[i]

    def _write_table_header(o):
        _write_line(o, ".. list-table::")
        _write_line(o, indent + ":widths: 25,50")
        _write_empty_line(o)

    def _write_submodules_summary_table(o, mod):
        _write_table_header(o)
        for submod in iter_modules(mod.__path__):
            if submod.ispkg:
                _write_line(
                    o,
                    indent
                    + "* - :ref:`"
                    + mod.__name__
                    + "."
                    + submod.name
                    + "_pyapi`",
                )
                _submod = import_module(
                    module + "." + submod.name, mod.__name__
                )
                mod_summary = _safe_get_docs(_submod)
                _write_line(o, indent + "  - " + mod_summary)
        _write_empty_line(o)

    def _write_classes_summary_table(o, mod):
        _write_table_header(o)
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and not (
                issubclass(obj, enum.Enum) or issubclass(obj, Exception)
            ):
                _write_line(o, indent + "* - :class:`" + obj.__name__ + "`")
                # For classes, the first line of the docstring is the
                # signature. So we skip that line to pick up the summary.
                cls_summary = _safe_get_docs(obj, 1)
                _write_line(o, indent + "  - " + cls_summary)
        _write_empty_line(o)

    def _write_enums_summary_table(o, mod):
        _write_table_header(o)
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and issubclass(obj, enum.Enum):
                _write_line(o, indent + "* - :class:`" + obj.__name__ + "`")
                enum_summary = _safe_get_docs(obj)
                _write_line(o, indent + "  - " + enum_summary)
        _write_empty_line(o)

    def _write_exceptions_summary_table(o, mod):
        _write_table_header(o)
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and issubclass(obj, Exception):
                _write_line(o, indent + "* - :class:`" + obj.__name__ + "`")
                # For classes, the first line of the docstring is the
                # signature. So we skip that line to pick up the summary.
                excp_summary = _safe_get_docs(obj, 1)
                _write_line(o, indent + "  - " + excp_summary)
        _write_empty_line(o)

    def _write_functions_summary_table(o, mod, fnobj_list):
        _write_table_header(o)
        for fnobj in fnobj_list:
            _write_line(o, indent + "* - :func:`" + fnobj.__name__ + "()`")
            # For functions, the first line of the docstring is the
            # signature. So we skip that line to pick up the summary.
            fn_summary = _safe_get_docs(fnobj, 1)
            _write_line(o, indent + "  - " + fn_summary)
        _write_empty_line(o)

    def _write_function_groups_summary(o, mod, groups):
        for group in groups:
            _write_empty_line(o)
            _write_underlined(o, group, "-")
            _write_empty_line(o)
            _write_functions_summary_table(o, mod, groups[group])

    mod = _get_module(module)

    with io.StringIO() as output:
        _write_line(output, rst_header)
        _write_empty_line(output)
        _write_marquee(output, pagename)
        _write_empty_line(output)
        _write_line(output, ".. currentmodule:: " + module)
        _write_empty_line(output)
        _write_line(output, ".. automodule:: " + module)
        _write_empty_line(output)
        _write_underlined(output, "Sub-modules", "-")
        _write_empty_line(output)
        _write_submodules_summary_table(output, mod)
        _write_underlined(output, "Classes", "-")
        _write_empty_line(output)
        _write_classes_summary_table(output, mod)
        _write_underlined(output, "Enums", "-")
        _write_empty_line(output)
        _write_enums_summary_table(output, mod)
        _write_underlined(output, "Exceptions", "-")
        _write_empty_line(output)
        _write_exceptions_summary_table(output, mod)
        _write_function_groups_summary(output, mod, _group_functions(mod))

        return output.getvalue()


def generate_rst_for_all_classes(module, outputpath):
    """Generates rst API docs for all classes in a module and writes them to
    given path.

    Args:
        module ([str]): Name of module that needs to be documented
        outputpath ([str]): Path where the rst files are to be saved.
    """
    mod = _get_module(module)

    if not os.path.exists(outputpath):
        raise ValueError("Invalid output path provided")
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and not (
            issubclass(obj, enum.Enum) or issubclass(obj, Exception)
        ):
            out = outputpath + "/" + name + ".rst"
            with open(out, "w") as rst_file:
                rst_file.write(generate_class_rst(obj))


def generate_rst_for_all_functions(module, outputpath):
    mod = _get_module(module)
    groups = _group_functions(mod)

    rst_header = "".join([".. _", module, "_functions_api:"])
    pagename = module + " Functions"

    if not os.path.exists(outputpath):
        raise ValueError("Invalid output path provided")

    def _write_function_autodocs(o, groups):
        for group, fnlist in groups.items():
            _write_empty_line(o)
            _write_underlined(o, group, "-")
            _write_empty_line(o)
            for fn in fnlist:
                _write_line(output, ".. autofunction:: " + fn.__name__)

    out = outputpath + "/" + module + "_functions_api.rst"
    with open(out, "w") as rst_file:
        with io.StringIO() as output:
            _write_line(output, rst_header)
            _write_empty_line(output)
            _write_marquee(output, pagename)
            _write_empty_line(output)
            _write_empty_line(output)
            _write_line(output, ".. currentmodule:: " + module)
            _write_empty_line(output)
            _write_function_autodocs(output, groups)
            rst_file.write(output.getvalue())


def generate_rst_for_all_exceptions(module, outputpath):
    mod = _get_module(module)
    rst_header = "".join([".. _", module, "_exception_api:"])
    pagename = module + " Exceptions"

    if not os.path.exists(outputpath):
        raise ValueError("Invalid output path provided")

    out = outputpath + "/" + module + "_exception_api.rst"
    with open(out, "w") as rst_file:
        with io.StringIO() as output:
            _write_line(output, rst_header)
            _write_empty_line(output)
            _write_marquee(output, pagename)
            _write_empty_line(output)
            _write_empty_line(output)
            _write_line(output, ".. currentmodule:: " + module)
            _write_empty_line(output)
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, Exception):
                    _write_line(output, ".. autoexception:: " + obj.__name__)

            rst_file.write(output.getvalue())


def generate_rst_for_all_enums(module, outputpath):
    mod = _get_module(module)
    indent = "    "
    rst_header = "".join([".. _", module, "_enum_api:"])
    pagename = module + " Enums"

    if not os.path.exists(outputpath):
        raise ValueError("Invalid output path provided")

    out = outputpath + "/" + module + "_enum_api.rst"
    with open(out, "w") as rst_file:
        with io.StringIO() as output:
            _write_line(output, rst_header)
            _write_empty_line(output)
            _write_marquee(output, pagename)
            _write_empty_line(output)
            _write_empty_line(output)
            _write_line(output, ".. currentmodule:: " + module)
            _write_empty_line(output)
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, enum.Enum):
                    _write_line(output, ".. autoclass:: " + obj.__name__)
                    _write_line(output, indent + ":members:")

            rst_file.write(output.getvalue())


def generate_all(module, outputpath):
    mod = _get_module(module)
    out = outputpath + "/" + module + "_pyapi.rst"
    # Generate a summary page for the module's API
    with open(out, "w") as rst_file:
        rst_file.write(generate_module_summary_rst(module))
    # Generate supporting pages for the module
    generate_rst_for_all_classes(module, outputpath)
    generate_rst_for_all_enums(module, outputpath)
    generate_rst_for_all_exceptions(module, outputpath)
    generate_rst_for_all_functions(module, outputpath)

    # Now recurse into any submodule and generate all for them too.
    for submod in iter_modules(mod.__path__):
        if submod.ispkg:
            generate_all(module + "." + submod.name, outputpath)
