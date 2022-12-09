#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2022 Intel Corporation
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

import contextlib
import operator

import numpy as np

import dpctl.tensor as dpt

__doc__ = (
    "Implementation module for printing " ":class:`dpctl.tensor.usm_ndarray`."
)

_print_options = {
    "linewidth": 75,
    "edgeitems": 3,
    "threshold": 1000,
    "precision": 8,
    "floatmode": "maxprec",
    "suppress": False,
    "nanstr": "nan",
    "infstr": "inf",
    "sign": "-",
}


def _options_dict(
    linewidth=None,
    edgeitems=None,
    threshold=None,
    precision=None,
    floatmode=None,
    suppress=None,
    nanstr=None,
    infstr=None,
    sign=None,
    numpy=False,
):
    if numpy:
        numpy_options = np.get_printoptions()
        options = {k: numpy_options[k] for k in _print_options.keys()}
    else:
        options = _print_options.copy()

    if suppress:
        options["suppress"] = True

    local = dict(locals().items())
    for int_arg in ["linewidth", "precision", "threshold", "edgeitems"]:
        val = local[int_arg]
        if val is not None:
            options[int_arg] = operator.index(val)

    for str_arg in ["nanstr", "infstr"]:
        val = local[str_arg]
        if val is not None:
            if not isinstance(val, str):
                raise TypeError(
                    "`{}` ".format(str_arg) + "must be of `string` type."
                )
            options[str_arg] = val

    signs = ["-", "+", " "]
    if sign is not None:
        if sign not in signs:
            raise ValueError(
                "`sign` must be one of"
                + ", ".join("`{}`".format(s) for s in signs)
            )
        options["sign"] = sign

    floatmodes = ["fixed", "unique", "maxprec", "maxprec_equal"]
    if floatmode is not None:
        if floatmode not in floatmodes:
            raise ValueError(
                "`floatmode` must be one of"
                + ", ".join("`{}`".format(m) for m in floatmodes)
            )
        options["floatmode"] = floatmode

    return options


def set_print_options(
    linewidth=None,
    edgeitems=None,
    threshold=None,
    precision=None,
    floatmode=None,
    suppress=None,
    nanstr=None,
    infstr=None,
    sign=None,
    numpy=False,
):
    options = _options_dict(
        linewidth=linewidth,
        edgeitems=edgeitems,
        threshold=threshold,
        precision=precision,
        floatmode=floatmode,
        suppress=suppress,
        nanstr=nanstr,
        infstr=infstr,
        sign=sign,
        numpy=numpy,
    )
    _print_options.update(options)


def get_print_options():
    return _print_options.copy()


@contextlib.contextmanager
def print_options(*args, **kwargs):
    options = dpt.get_print_options()
    try:
        dpt.set_print_options(*args, **kwargs)
        yield dpt.get_print_options()
    finally:
        dpt.set_print_options(**options)


def _nd_corners(x, edge_items, slices=()):
    axes_reduced = len(slices)
    if axes_reduced == x.ndim:
        return x[slices]

    if x.shape[axes_reduced] > 2 * edge_items:
        return dpt.concat(
            (
                _nd_corners(
                    x, edge_items, slices + (slice(None, edge_items, None),)
                ),
                _nd_corners(
                    x, edge_items, slices + (slice(-edge_items, None, None),)
                ),
            ),
            axis=axes_reduced,
        )
    else:
        return _nd_corners(x, edge_items, slices + (slice(None, None, None),))


def _usm_ndarray_str(
    x,
    line_width=None,
    edge_items=None,
    threshold=None,
    precision=None,
    floatmode=None,
    suppress=None,
    sign=None,
    numpy=False,
    separator=" ",
    prefix="",
    suffix="",
):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    options = get_print_options()
    options.update(
        _options_dict(
            linewidth=line_width,
            edgeitems=edge_items,
            threshold=threshold,
            precision=precision,
            floatmode=floatmode,
            suppress=suppress,
            sign=sign,
            numpy=numpy,
        )
    )

    threshold = options["threshold"]
    edge_items = options["edgeitems"]

    if x.size > threshold:
        # need edge_items + 1 elements for np.array2string to abbreviate
        data = dpt.asnumpy(_nd_corners(x, edge_items + 1))
        options["threshold"] = 0
    else:
        data = dpt.asnumpy(x)
    with np.printoptions(**options):
        s = np.array2string(
            data, separator=separator, prefix=prefix, suffix=suffix
        )
    return s


def _usm_ndarray_repr(x, line_width=None, precision=None, suppress=None):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    if line_width is None:
        line_width = _print_options["linewidth"]

    show_dtype = x.dtype not in [
        dpt.bool,
        dpt.int64,
        dpt.float64,
        dpt.complex128,
    ]

    prefix = "usm_ndarray("
    suffix = ")"

    s = _usm_ndarray_str(
        x,
        line_width=line_width,
        precision=precision,
        suppress=suppress,
        separator=", ",
        prefix=prefix,
        suffix=suffix,
    )

    if show_dtype:
        dtype_str = "dtype={}".format(x.dtype.name)
        bottom_len = len(s) - (s.rfind("\n") + 1)
        next_line = bottom_len + len(dtype_str) + 1 > line_width
        dtype_str = ",\n" + dtype_str if next_line else ", " + dtype_str
    else:
        dtype_str = ""

    return prefix + s + dtype_str + suffix
