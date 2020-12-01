import numpy as np
from inspect import getmembers, isfunction, isclass, isbuiltin
from numbers import Number
from types import FunctionType as ftype, BuiltinFunctionType as bftype
import sys
#import importlib
#import functools
import inspect

debug = False

def dprint(*args):
    if debug:
        print(*args)
        sys.stdout.flush()

import dpctl
from dpctl.memory import MemoryUSMShared

functions_list = [o[0] for o in getmembers(np) if isfunction(o[1]) or isbuiltin(o[1])]
class_list = [o for o in getmembers(np) if isclass(o[1])]

array_interface_property = "__array_interface__"
def has_array_interface(x):
    return hasattr(x, array_interface_property)

class ndarray(np.ndarray):
    """
    numpy.ndarray subclass whose underlying memory buffer is allocated
    with a foreign allocator.
    """
    def __new__(subtype, shape,
                dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        # Create a new array.
        if buffer is None:
            dprint("dparray::ndarray __new__ buffer None")
            nelems = np.prod(shape)
            dt = np.dtype(dtype)
            isz = dt.itemsize
            nbytes = int(isz*max(1, nelems))
            buf = MemoryUSMShared(nbytes)
            new_obj = np.ndarray.__new__(
                subtype, shape, dtype=dt,
                buffer=buf, offset=0,
                strides=strides, order=order)
            if hasattr(new_obj, array_interface_property):
                dprint("buffer None new_obj already has sycl_usm")
            else:
                dprint("buffer None new_obj will add sycl_usm")
                setattr(new_obj, array_interface_property, {})
            return new_obj
        # zero copy if buffer is a usm backed array-like thing
        elif hasattr(buffer, array_interface_property):
            dprint("dparray::ndarray __new__ buffer", array_interface_property)
            # also check for array interface
            new_obj = np.ndarray.__new__(
                subtype, shape, dtype=dtype,
                buffer=buffer, offset=offset,
                strides=strides, order=order)
            if hasattr(new_obj, array_interface_property):
                dprint("buffer None new_obj already has sycl_usm")
            else:
                dprint("buffer None new_obj will add sycl_usm")
                setattr(new_obj, array_interface_property, {})
            return new_obj
        else:
            dprint("dparray::ndarray __new__ buffer not None and not sycl_usm")
            nelems = np.prod(shape)
            # must copy
            ar = np.ndarray(shape,
                            dtype=dtype, buffer=buffer,
                            offset=offset, strides=strides,
                            order=order)
            nbytes = int(ar.nbytes)
            buf = MemoryUSMShared(nbytes)
            new_obj = np.ndarray.__new__(
                subtype, shape, dtype=dtype,
                buffer=buf, offset=0,
                strides=strides, order=order)
            np.copyto(new_obj, ar, casting='no')
            if hasattr(new_obj, array_interface_property):
                dprint("buffer None new_obj already has sycl_usm")
            else:
                dprint("buffer None new_obj will add sycl_usm")
                setattr(new_obj, array_interface_property, {})
            return new_obj

    def __array_finalize__(self, obj):
        dprint("__array_finalize__:", obj, hex(id(obj)), type(obj))
        # When called from the explicit constructor, obj is None
        if obj is None: return
        # When called in new-from-template, `obj` is another instance of our own
        # subclass, that we might use to update the new `self` instance.
        # However, when called from view casting, `obj` can be an instance of any
        # subclass of ndarray, including our own.
        if hasattr(obj, array_interface_property):
            return
        if isinstance(obj, numba.core.runtime._nrt_python._MemInfo):
            mobj = obj
            while isinstance(mobj, numba.core.runtime._nrt_python._MemInfo):
                dprint("array_finalize got Numba MemInfo")
                ea = mobj.external_allocator
                d = mobj.data
                dprint("external_allocator:", hex(ea), type(ea))
                dprint("data:", hex(d), type(d))
                dppl_rt_allocator = numba.dppl._dppl_rt.get_external_allocator()
                dprint("dppl external_allocator:", hex(dppl_rt_allocator), type(dppl_rt_allocator))
                dprint(dir(mobj))
                if ea == dppl_rt_allocator:
                    return
                mobj = mobj.parent
                if isinstance(mobj, ndarray):
                    mobj = mobj.base
        if isinstance(obj, np.ndarray):
            ob = self
            while isinstance(ob, np.ndarray):
                if hasattr(obj, array_interface_property):
                    return
                ob = ob.base
    
        # Just raise an exception since __array_ufunc__ makes all reasonable cases not
        # need the code below.
        raise ValueError("Non-USM allocated ndarray can not viewed as a USM-allocated one without a copy")
      
    # Tell Numba to not treat this type just like a NumPy ndarray but to propagate its type.
    # This way it will use the custom dparray allocator.
    __numba_no_subtype_ndarray__ = True

    # Convert to a NumPy ndarray.
    def as_ndarray(self):
         return np.copy(self)

    def __array__(self):
        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
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
            # If no out kwarg then we create a dparray out so that we get
            # USM memory.  However, if kwarg has dparray-typed out then
            # array_ufunc is called recursively so we cast out as regular
            # NumPy ndarray (having a USM data pointer).
            if kwargs.get('out', None) is None:
                # maybe copy?
                # deal with multiple returned arrays, so kwargs['out'] can be tuple
                res_type = np.result_type(*typing)
                out = empty(inputs[0].shape, dtype=res_type)
                out_as_np = np.ndarray(out.shape, out.dtype, out)
                kwargs['out'] = out_as_np
            else:
                # If they manually gave dparray as out kwarg then we have to also
                # cast as regular NumPy ndarray to avoid recursion.
                if isinstance(kwargs['out'], ndarray):
                    out = kwargs['out']
                    kwargs['out'] = np.ndarray(out.shape, out.dtype, out)
                else:
                    out = kwargs['out']
            ret = ufunc(*scalars, **kwargs)
            return out
        else:
            return NotImplemented

def isdef(x):
    try:
        eval(x)
        return True
    except NameError:
        return False

for c in class_list:
    cname = c[0]
    if isdef(cname):
        continue
    # For now we do the simple thing and copy the types from NumPy module into dparray module.
    new_func = "%s = np.%s" % (cname, cname)
    try:
        the_code = compile(new_func, '__init__', 'exec')
        exec(the_code)
    except:
        print("Failed to exec type propagation", cname)
        pass

# Redefine all Numpy functions in this module and if they
# return a Numpy array, transform that to a USM-backed array
# instead.  This is a stop-gap.  We should eventually find a
# way to do the allocation correct to start with.
for fname in functions_list:
    if isdef(fname):
        continue
    new_func =  "def %s(*args, **kwargs):\n" % fname
    new_func += "    ret = np.%s(*args, **kwargs)\n" % fname
    new_func += "    if type(ret) == np.ndarray:\n"
    new_func += "        ret = ndarray(ret.shape, ret.dtype, ret)\n"
    new_func += "    return ret\n"
    the_code = compile(new_func, '__init__', 'exec')
    exec(the_code)

def from_ndarray(x):
    return copy(x)

def as_ndarray(x):
     return np.copy(x)
