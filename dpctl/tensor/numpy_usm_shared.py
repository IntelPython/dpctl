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

"""Provides an implementation of a SYCL malloc_shared allocator-aware Numpy.

The modules includes a ``numpy.ndarray`` sub-class whose underlying memory
buffer is allocated with a SYCL malloc_shared memory allocator. The
``malloc_shared`` allocated array is meant to be interoperable with Python
extension modules that use SYCL and support USM.

The module also includes all NumPy classes and functions and the module can be
used as a drop-in replacement for standard NumPy. Note that this module does
not include any kernel implementations and has the same kernels as standard
NumPy.

"""

import builtins
import sys
from inspect import getmembers, isbuiltin, isclass, isfunction
from numbers import Number

import numpy as np

from dpctl.memory import MemoryUSMShared

debug = False


def dprint(*args):
    if debug:
        print(*args)
        sys.stdout.flush()


functions_list = []
class_list = []
for o in getmembers(np):
    s = o[1]
    if isfunction(s) or isbuiltin(s):
        functions_list.append(o[0])
    elif isclass(s):
        class_list.append(o)


array_interface_property = "__sycl_usm_array_interface__"


def has_array_interface(x):
    return hasattr(x, array_interface_property)


def _get_usm_base(ary):
    ob = ary
    while True:
        if ob is None:
            return None
        elif hasattr(ob, "__sycl_usm_array_interface__"):
            return ob
        elif isinstance(ob, np.ndarray):
            ob = ob.base
        elif isinstance(ob, memoryview):
            ob = ob.obj
        else:
            return None


def convert_ndarray_to_np_ndarray(x, require_ndarray=False):
    if isinstance(x, ndarray):
        return np.array(x, copy=False, subok=False)
    elif isinstance(x, tuple):
        return tuple(
            convert_ndarray_to_np_ndarray(y, require_ndarray=require_ndarray)
            for y in x
        )
    elif require_ndarray:
        raise TypeError
    else:
        return x


class ndarray(np.ndarray):
    """
    numpy.ndarray subclass whose underlying memory buffer is allocated
    with a foreign allocator.
    """

    external_usm_checkers = []

    def add_external_usm_checker(func):
        ndarray.external_usm_checkers.append(func)

    def __new__(
        subtype,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
    ):
        # Create a new array.
        if buffer is None:
            dprint("numpy_usm_shared::ndarray __new__ buffer None")
            nelems = np.prod(shape)
            dt = np.dtype(dtype)
            isz = dt.itemsize
            # Have to use builtins.max explicitly since this module will
            # import numpy's max function.
            nbytes = int(isz * builtins.max(1, nelems))
            buf = MemoryUSMShared(nbytes)
            new_obj = np.ndarray.__new__(
                subtype,
                shape,
                dtype=dt,
                buffer=buf,
                offset=0,
                strides=strides,
                order=order,
            )
            if hasattr(new_obj, array_interface_property):
                dprint("buffer None new_obj already has sycl_usm")
            else:
                dprint("buffer None new_obj will add sycl_usm")
                setattr(
                    new_obj,
                    array_interface_property,
                    new_obj._getter_sycl_usm_array_interface_(),
                )
            return new_obj
        # zero copy if buffer is a usm backed array-like thing
        elif hasattr(buffer, array_interface_property):
            dprint(
                "numpy_usm_shared::ndarray __new__ buffer",
                array_interface_property,
            )
            # also check for array interface
            new_obj = np.ndarray.__new__(
                subtype,
                shape,
                dtype=dtype,
                buffer=buffer,
                offset=offset,
                strides=strides,
                order=order,
            )
            if hasattr(new_obj, array_interface_property):
                dprint("buffer None new_obj already has sycl_usm")
            else:
                dprint("buffer None new_obj will add sycl_usm")
                setattr(
                    new_obj,
                    array_interface_property,
                    new_obj._getter_sycl_usm_array_interface_(),
                )
            return new_obj
        else:
            dprint(
                "numpy_usm_shared::ndarray __new__ buffer not None "
                "and not sycl_usm"
            )
            nelems = np.prod(shape)
            # must copy
            ar = np.ndarray(
                shape,
                dtype=dtype,
                buffer=buffer,
                offset=offset,
                strides=strides,
                order=order,
            )
            nbytes = int(ar.nbytes)
            buf = MemoryUSMShared(nbytes)
            new_obj = np.ndarray.__new__(
                subtype,
                shape,
                dtype=dtype,
                buffer=buf,
                offset=0,
                strides=strides,
                order=order,
            )
            np.copyto(new_obj, ar, casting="no")
            if hasattr(new_obj, array_interface_property):
                dprint("buffer None new_obj already has sycl_usm")
            else:
                dprint("buffer None new_obj will add sycl_usm")
                setattr(
                    new_obj,
                    array_interface_property,
                    new_obj._getter_sycl_usm_array_interface_(),
                )
            return new_obj

    def __sycl_usm_array_interface__(self):
        return self._getter_sycl_usm_array_interface()

    def _getter_sycl_usm_array_interface_(self):
        ary_iface = self.__array_interface__
        _base = _get_usm_base(self)
        if _base is None:
            raise TypeError

        usm_iface = getattr(_base, "__sycl_usm_array_interface__", None)
        if usm_iface is None:
            raise TypeError

        if ary_iface["data"][0] == usm_iface["data"][0]:
            ary_iface["version"] = usm_iface["version"]
            ary_iface["syclobj"] = usm_iface["syclobj"]
        else:
            raise TypeError
        return ary_iface

    def __array_finalize__(self, obj):
        dprint("__array_finalize__:", obj, hex(id(obj)), type(obj))
        # When called from the explicit constructor, obj is None
        if obj is None:
            return
        # When called in new-from-template, `obj` is another instance of our own
        # subclass, that we might use to update the new `self` instance.
        # However, when called from view casting, `obj` can be an instance of
        # any subclass of ndarray, including our own.
        if hasattr(obj, array_interface_property):
            return
        for ext_checker in ndarray.external_usm_checkers:
            if ext_checker(obj):
                return
        if isinstance(obj, np.ndarray):
            ob = self
            while isinstance(ob, np.ndarray):
                if hasattr(ob, array_interface_property):
                    return
                ob = ob.base

        # Just raise an exception since __array_ufunc__ makes all
        # reasonable cases not need the code below.
        raise ValueError(
            "Non-USM allocated ndarray can not viewed as a USM-allocated "
            "one without a copy."
        )

    # Tell Numba to not treat this type just like a NumPy ndarray but to
    # propagate its type. This way it will use the custom numpy_usm_shared
    # allocator.
    __numba_no_subtype_ndarray__ = True

    # Convert to a NumPy ndarray.
    def as_ndarray(self):
        return np.array(self, copy=True, subok=False)

    def __array__(self):
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            N = None
            scalars = []
            typing = []
            for inp in inputs:
                if isinstance(inp, Number):
                    scalars.append(inp)
                    typing.append(inp)
                elif isinstance(inp, (self.__class__, np.ndarray)):
                    if isinstance(inp, self.__class__):
                        scalars.append(np.ndarray(inp.shape, inp.dtype, inp))
                        typing.append(np.ndarray(inp.shape, inp.dtype))
                    else:
                        scalars.append(inp)
                        typing.append(inp)
                    if N is not None:
                        if N != inp.shape:
                            raise TypeError("inconsistent sizes")
                    else:
                        N = inp.shape
                else:
                    return NotImplemented
            # Have to avoid recursive calls to array_ufunc here.
            # If no out kwarg then we create a numpy_usm_shared out so that we
            # get USM memory.  However, if kwarg has numpy_usm_shared-typed out
            # then array_ufunc is called recursively so we cast out as regular
            # NumPy ndarray (having a USM data pointer).
            out_arg = kwargs.get("out", None)
            if out_arg is None:
                # maybe copy?
                # deal with multiple returned arrays, so kwargs['out'] can be
                # tuple.
                res_type = np.result_type(*typing)
                out_arg = empty(inputs[0].shape, dtype=res_type)
                out_as_np = convert_ndarray_to_np_ndarray(out_arg)
                kwargs["out"] = out_as_np
            else:
                # If they manually gave numpy_usm_shared as out kwarg then we
                # have to also cast as regular NumPy ndarray to avoid recursion.
                try:
                    kwargs["out"] = convert_ndarray_to_np_ndarray(
                        out_arg, require_ndarray=True
                    )
                except TypeError:
                    raise TypeError(
                        "Return arrays must each be {}".format(self.__class__)
                    )
            ufunc(*scalars, **kwargs)
            return out_arg
        elif method == "reduce":
            N = None
            scalars = []
            typing = []
            for inp in inputs:
                if isinstance(inp, Number):
                    scalars.append(inp)
                    typing.append(inp)
                elif isinstance(inp, (self.__class__, np.ndarray)):
                    if isinstance(inp, self.__class__):
                        scalars.append(np.ndarray(inp.shape, inp.dtype, inp))
                        typing.append(np.ndarray(inp.shape, inp.dtype))
                    else:
                        scalars.append(inp)
                        typing.append(inp)
                    if N is not None:
                        if N != inp.shape:
                            raise TypeError("inconsistent sizes")
                    else:
                        N = inp.shape
                else:
                    return NotImplemented
            assert "out" not in kwargs
            return super().__array_ufunc__(ufunc, method, *scalars, **kwargs)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        fname = func.__name__
        has_func = _isdef(fname)
        dprint("__array_function__:", func, fname, type(func), types, has_func)
        if has_func:
            cm = sys.modules[__name__]
            affunc = getattr(cm, fname)
            fargs = [
                x.view(np.ndarray) if isinstance(x, ndarray) else x
                for x in args
            ]
            fkwargs = {
                key: convert_ndarray_to_np_ndarray(val)
                for key, val in kwargs.items()
            }
            res = affunc(*fargs, **fkwargs)
            return kwargs["out"] if "out" in kwargs else res
        return NotImplemented


def _isdef(x):
    cm = sys.modules[__name__]
    return hasattr(cm, x)


for c in class_list:
    cname = c[0]
    if _isdef(cname):
        continue
    # For now we do the simple thing and copy the types from NumPy module
    # into numpy_usm_shared module.
    new_func = "%s = np.%s" % (cname, cname)
    try:
        the_code = compile(new_func, "__init__", "exec")
        exec(the_code)
    except Exception:
        print("Failed to exec type propagation", cname)
        pass

# Redefine all Numpy functions in this module and if they
# return a Numpy array, transform that to a USM-backed array
# instead.  This is a stop-gap.  We should eventually find a
# way to do the allocation correct to start with.
for fname in functions_list:
    if _isdef(fname):
        continue
    new_func = "def %s(*args, **kwargs):\n" % fname
    new_func += "    ret = np.%s(*args, **kwargs)\n" % fname
    new_func += "    if type(ret) == np.ndarray:\n"
    new_func += "        ret = ndarray(ret.shape, ret.dtype, ret)\n"
    new_func += "    return ret\n"
    the_code = compile(new_func, "__init__", "exec")
    exec(the_code)


def from_ndarray(x):
    return copy(x)


def as_ndarray(x):
    return np.copy(np.ndarray(x.shape, x.dtype, x))
