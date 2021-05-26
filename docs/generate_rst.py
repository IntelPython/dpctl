import inspect
import io

import dpctl

wrapper_descriptor = type(dpctl.SyclDevice.__init__)

# known property in Cython extension class
getset_descriptor = type(dpctl.SyclDevice.name)
# known method (defined using def in Cython extension class)
cython_method_type = type(dpctl.SyclDevice.get_filter_string)
# known builtin method (defined using cpdef in Cython extension class)
cython_builtin_function_or_method_type = type(dpctl.SyclQueue.mro)


def get_public_class_name(cls):
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


def is_class_property(o):
    return isinstance(o, property) or (type(o) == getset_descriptor)


def is_class_method(o):
    return inspect.ismethod(o) or (
        type(o) in [cython_method_type, cython_builtin_function_or_method_type]
    )


def get_filtered_names(cls, selector_func):
    return [
        _name
        for _name, _obj in inspect.getmembers(cls, selector_func)
        if not _name.startswith("__")
    ]


def generate_class_rst(cls):
    if not inspect.isclass(cls):
        raise TypeError("Expecting class, got {}".format(type(cls)))

    cls_qualname = get_public_class_name(cls)
    # cls_property_names = get_filtered_names(cls, is_class_property)
    # cls_method_names = get_filtered_names(cls, is_class_method)
    rst_header = cls_qualname.split(".")[-1]
    rst_module = ".".join(cls_qualname.split(".")[:-1])
    rst_header = "".join([".. _", rst_header, "_api:"])
    empty_line = ""

    def write_line(o, s):
        o.write(s)
        o.write("\n")

    def write_rubric(o, indent, rubric_display, rubric_tag, cls_qualname):
        write_line(o, indent + ".. rubric:: " + rubric_display)
        write_line(o, "")
        write_line(o, indent + ".. autosummary:: " + cls_qualname)
        write_line(o, indent + indent + ":" + rubric_tag + ":")
        write_line(o, "")

    def write_underlined(o, s, c):
        write_line(o, s)
        write_line(o, c * len(s))

    with io.StringIO() as output:
        write_line(output, rst_header)
        write_line(output, empty_line)
        marquee = "#" * len(cls_qualname)
        write_line(output, marquee)
        write_line(output, cls_qualname)
        write_line(output, marquee)
        write_line(output, empty_line)

        write_line(output, ".. currentmodule:: " + rst_module)
        write_line(output, empty_line)

        write_line(output, ".. autoclass:: " + cls_qualname)
        write_line(output, empty_line)

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

        write_underlined(output, "Detail", "=")
        write_line(output, empty_line)

        # Attributes
        all_attributes = get_filtered_names(cls, is_class_property)
        if all_attributes:
            write_underlined(output, attributes_header, "-")
            write_line(output, empty_line)
            for n in all_attributes:
                write_line(
                    output,
                    ".. autoattribute:: " + ".".join([cls_qualname, n]),
                )
            write_line(output, empty_line)

        # Methods, separated into public/private
        all_methods = get_filtered_names(cls, is_class_method)
        all_public_methods = []
        all_private_methods = []
        for _name in all_methods:
            if _name.startswith("_"):
                all_public_methods.append(_name)
            else:
                all_private_methods.append(_name)

        if all_public_methods:
            write_underlined(output, public_methods_header, "-")
            write_line(output, empty_line)
            for n in all_public_methods:
                write_line(
                    output,
                    ".. autoattribute:: " + ".".join([cls_qualname, n]),
                )
            write_line(output, empty_line)

        # Private methods
        if all_private_methods:
            write_underlined(output, private_methods_header, "-")
            write_line(output, empty_line)
            for n in all_private_methods:
                write_line(
                    output,
                    ".. autoattribute:: " + ".".join([cls_qualname, n]),
                )
            write_line(output, empty_line)

        return output.getvalue()
