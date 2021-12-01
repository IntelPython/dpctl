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
                all_public_methods.append(_name)
            else:
                all_private_methods.append(_name)

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


def generate_landing_rst(module):
    """[summary]

    Args:
        module ([str]): [description]

    Returns:
        [type]: [description]
    """
    rst_header = "".join([".. _", module, "_pyapi:"])
    pagename = module + " API"
    indent = "    "

    def _write_table_header(o):
        _write_line(o, ".. list-table::")
        _write_line(o, indent + ":widths: 25,50")
        _write_empty_line(o)

    def _write_submodule_table(o, mod):
        _write_table_header(o)
        for submod in iter_modules(mod.__path__):
            if submod.ispkg:
                _write_line(
                    o,
                    indent
                    + "* - :ref:`"
                    + module
                    + "."
                    + submod.name
                    + "_api`",
                )
                _submod = import_module(
                    module + "." + submod.name, mod.__name__
                )
                mod_summary = (
                    ""
                    if not _submod.__doc__
                    else _submod.__doc__.split("\n")[0]
                )
                _write_line(o, indent + "  - " + mod_summary)
        _write_empty_line(o)

    def _write_classes_table(o, mod):
        _write_table_header(o)
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and not (
                issubclass(obj, enum.Enum) or issubclass(obj, Exception)
            ):
                _write_line(o, indent + "* - :ref:`" + name + "_api`")
                # For classes, the first line of the docstring is the
                # signature. So we skip that line to pick up the summary.
                cls_summary = obj.__doc__.split("\n")[1]
                _write_line(o, indent + "  - " + cls_summary)
        _write_empty_line(o)

    def _write_enum_table(o, mod):
        _write_table_header(o)
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and issubclass(obj, enum.Enum):
                # FIXME link into the page pointing to the actual doc
                # section for the enum.
                _write_line(o, indent + "* - :ref:`" + module + "_enum_api`")
                enum_summary = obj.__doc__.split("\n")[0]
                _write_line(o, indent + "  - " + enum_summary)
        _write_empty_line(o)

    def _write_exception_table(o, mod):
        _write_table_header(o)
        for name, obj in inspect.getmembers(mod):
            if inspect.isclass(obj) and issubclass(obj, Exception):
                _write_line(
                    o, indent + "* - :ref:`" + module + "_exception_api`"
                )
                # For classes, the first line of the docstring is the
                # signature. So we skip that line to pick up the summary.
                excp_summary = obj.__doc__.split("\n")[1]
                _write_line(o, indent + "  - " + excp_summary)
        _write_empty_line(o)

    mod = None
    try:
        mod = sys.modules[module]
    except KeyError:
        raise ValueError(
            module + "is not a valid module name or it is not loaded"
        )

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
        _write_submodule_table(output, mod)
        _write_underlined(output, "Classes", "-")
        _write_empty_line(output)
        _write_classes_table(output, mod)
        _write_enum_table(output, mod)
        _write_exception_table(output, mod)
        return output.getvalue()


def generate_rst_for_all_classes(module, outputpath):
    """Generates rst API docs for all classes in a module and writes them to
    given path.

    Args:
        module ([str]): Name of module that needs to be documented
        outputpath ([str]): Path where the rst files are to be saved.
    """
    mod = None
    try:
        mod = sys.modules[module]
    except KeyError:
        raise ValueError(
            module + "is not a valid module name or it is not loaded"
        )

    if not os.path.exists(outputpath):
        raise ValueError("Invalid output path provided")
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and not issubclass(obj, enum.Enum):
            out = outputpath + "/" + name + ".rst"
            with open(out, "w") as rst_file:
                rst_file.write(generate_class_rst(obj))
