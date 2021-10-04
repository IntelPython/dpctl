#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import numbers

import numpy as np
import numpy.lib.stride_tricks as np_st
import pytest

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
from dpctl.tensor._usmarray import Device


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (4,),
        (0,),
        (0, 1),
        (0, 0),
        (4, 5),
        (2, 5, 2),
        (2, 2, 2, 2, 2, 2, 2, 2),
        5,
    ],
)
@pytest.mark.parametrize("usm_type", ["shared", "host", "device"])
def test_allocate_usm_ndarray(shape, usm_type):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclCreationError:
        pytest.skip("Default SYCL queue could not be created")
    X = dpt.usm_ndarray(
        shape, dtype="d", buffer=usm_type, buffer_ctor_kwargs={"queue": q}
    )
    Xnp = np.ndarray(shape, dtype="d")
    assert X.usm_type == usm_type
    assert X.sycl_context == q.sycl_context
    assert X.sycl_device == q.sycl_device
    assert X.size == Xnp.size
    assert X.shape == Xnp.shape
    assert X.shape == X.__sycl_usm_array_interface__["shape"]


@pytest.mark.parametrize(
    "dtype",
    [
        "u1",
        "i1",
        "u2",
        "i2",
        "u4",
        "i4",
        "u8",
        "i8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
        b"float32",
        np.dtype("d"),
        np.half,
    ],
)
def test_dtypes(dtype):
    Xusm = dpt.usm_ndarray((1,), dtype=dtype)
    assert Xusm.itemsize == np.dtype(dtype).itemsize
    expected_fmt = (np.dtype(dtype).str)[1:]
    actual_fmt = Xusm.__sycl_usm_array_interface__["typestr"][1:]
    assert expected_fmt == actual_fmt


@pytest.mark.parametrize("dtype", ["", ">f4", "invalid", 123])
def test_dtypes_invalid(dtype):
    with pytest.raises((TypeError, ValueError)):
        dpt.usm_ndarray((1,), dtype=dtype)


@pytest.mark.parametrize("dt", ["d", "c16"])
def test_properties(dt):
    """
    Test that properties execute
    """
    X = dpt.usm_ndarray((3, 4, 5), dtype=dt)
    assert isinstance(X.sycl_queue, dpctl.SyclQueue)
    assert isinstance(X.sycl_device, dpctl.SyclDevice)
    assert isinstance(X.sycl_context, dpctl.SyclContext)
    assert isinstance(X.dtype, np.dtype)
    assert isinstance(X.__sycl_usm_array_interface__, dict)
    assert isinstance(X.T, dpt.usm_ndarray)
    assert isinstance(X.imag, dpt.usm_ndarray)
    assert isinstance(X.real, dpt.usm_ndarray)
    assert isinstance(X.shape, tuple)
    assert isinstance(X.strides, tuple)
    assert X.usm_type in ("shared", "device", "host")
    assert isinstance(X.size, numbers.Integral)
    assert isinstance(X.nbytes, numbers.Integral)
    assert isinstance(X.ndim, numbers.Integral)
    assert isinstance(X._pointer, numbers.Integral)


@pytest.mark.parametrize("func", [bool, float, int, complex])
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("dtype", ["|b1", "|u2", "|f4", "|i8"])
def test_copy_scalar_with_func(func, shape, dtype):
    X = dpt.usm_ndarray(shape, dtype=dtype)
    Y = np.arange(1, X.size + 1, dtype=dtype).reshape(shape)
    X.usm_data.copy_from_host(Y.reshape(-1).view("|u1"))
    assert func(X) == func(Y)


@pytest.mark.parametrize(
    "method", ["__bool__", "__float__", "__int__", "__complex__"]
)
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("dtype", ["|b1", "|u2", "|f4", "|i8"])
def test_copy_scalar_with_method(method, shape, dtype):
    X = dpt.usm_ndarray(shape, dtype=dtype)
    Y = np.arange(1, X.size + 1, dtype=dtype).reshape(shape)
    X.usm_data.copy_from_host(Y.reshape(-1).view("|u1"))
    assert getattr(X, method)() == getattr(Y, method)()


@pytest.mark.parametrize("func", [bool, float, int, complex])
@pytest.mark.parametrize("shape", [(2,), (1, 2), (3, 4, 5), (0,)])
def test_copy_scalar_invalid_shape(func, shape):
    X = dpt.usm_ndarray(shape)
    with pytest.raises(ValueError):
        func(X)


def test_index_noninteger():
    import operator

    X = dpt.usm_ndarray(1, "d")
    with pytest.raises(IndexError):
        operator.index(X)


@pytest.mark.parametrize(
    "ind",
    [
        tuple(),
        (None,),
        (
            None,
            Ellipsis,
            None,
        ),
        (2, 2, None, 3, 4),
        (Ellipsis,),
        (None, slice(0, None, 2), Ellipsis, slice(0, None, 3)),
        (None, slice(1, None, 2), Ellipsis, slice(1, None, 3)),
        (None, slice(None, -1, -2), Ellipsis, slice(2, None, 3)),
        (
            slice(None, None, -1),
            slice(None, None, -1),
            slice(0, None, 3),
            slice(1, None, 2),
        ),
    ],
)
def test_basic_slice(ind):
    X = dpt.usm_ndarray((2 * 3, 2 * 4, 3 * 5, 2 * 7), dtype="u1")
    Xnp = np.empty(X.shape, dtype=X.dtype)
    S = X[ind]
    Snp = Xnp[ind]
    assert S.shape == Snp.shape
    assert S.strides == Snp.strides
    assert S.dtype == X.dtype


def _from_numpy(np_ary, device=None, usm_type="shared"):
    if type(np_ary) is np.ndarray:
        if np_ary.flags["FORC"]:
            x = np_ary
        else:
            x = np.ascontiguous(np_ary)
        R = dpt.usm_ndarray(
            np_ary.shape,
            dtype=np_ary.dtype,
            buffer=usm_type,
            buffer_ctor_kwargs={
                "queue": Device.create_device(device).sycl_queue
            },
        )
        R.usm_data.copy_from_host(x.reshape((-1)).view("|u1"))
        return R
    else:
        raise ValueError("Expected numpy.ndarray, got {}".format(type(np_ary)))


def _to_numpy(usm_ary):
    if type(usm_ary) is dpt.usm_ndarray:
        usm_buf = usm_ary.usm_data
        s = usm_buf.nbytes
        host_buf = usm_buf.copy_to_host().view(usm_ary.dtype)
        usm_ary_itemsize = usm_ary.itemsize
        R_offset = (
            usm_ary.__sycl_usm_array_interface__["offset"] * usm_ary_itemsize
        )
        R = np.ndarray((s,), dtype="u1", buffer=host_buf)
        R = R[R_offset:].view(usm_ary.dtype)
        R_strides = (usm_ary_itemsize * si for si in usm_ary.strides)
        return np_st.as_strided(R, shape=usm_ary.shape, strides=R_strides)
    else:
        raise ValueError(
            "Expected dpctl.tensor.usm_ndarray, got {}".format(type(usm_ary))
        )


def test_slice_constructor_1d():
    Xh = np.arange(37, dtype="i4")
    default_device = dpctl.select_default_device()
    Xusm = _from_numpy(Xh, device=default_device, usm_type="device")
    for ind in [
        slice(1, None, 2),
        slice(0, None, 3),
        slice(1, None, 3),
        slice(2, None, 3),
        slice(None, None, -1),
        slice(-2, 2, -2),
        slice(-1, 1, -2),
        slice(None, None, -13),
    ]:
        assert np.array_equal(
            _to_numpy(Xusm[ind]), Xh[ind]
        ), "Failed for {}".format(ind)


def test_slice_constructor_3d():
    Xh = np.empty((37, 24, 35), dtype="i4")
    default_device = dpctl.select_default_device()
    Xusm = _from_numpy(Xh, device=default_device, usm_type="device")
    for ind in [
        slice(1, None, 2),
        slice(0, None, 3),
        slice(1, None, 3),
        slice(2, None, 3),
        slice(None, None, -1),
        slice(-2, 2, -2),
        slice(-1, 1, -2),
        slice(None, None, -13),
        (slice(None, None, -2), Ellipsis, None, 15),
    ]:
        assert np.array_equal(
            _to_numpy(Xusm[ind]), Xh[ind]
        ), "Failed for {}".format(ind)


@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_slice_suai(usm_type):
    Xh = np.arange(0, 10, dtype="u1")
    default_device = dpctl.select_default_device()
    Xusm = _from_numpy(Xh, device=default_device, usm_type=usm_type)
    for ind in [slice(2, 3, None), slice(5, 7, None), slice(3, 9, None)]:
        assert np.array_equal(
            dpm.as_usm_memory(Xusm[ind]).copy_to_host(), Xh[ind]
        ), "Failed for {}".format(ind)


def test_slicing_basic():
    Xusm = dpt.usm_ndarray((10, 5), dtype="c16")
    Xusm[None]
    Xusm[...]
    Xusm[8]
    Xusm[-3]
    with pytest.raises(IndexError):
        Xusm[..., ...]
    with pytest.raises(IndexError):
        Xusm[1, 1, :, 1]
    Xusm[:, -4]
    with pytest.raises(IndexError):
        Xusm[:, -128]
    with pytest.raises(TypeError):
        Xusm[{1, 2, 3, 4, 5, 6, 7}]
    X = dpt.usm_ndarray(10, "u1")
    X.usm_data.copy_from_host(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")
    int(
        X[X[2]]
    )  # check that objects with __index__ method can be used as indices
    Xh = dpm.as_usm_memory(X[X[2] : X[5]]).copy_to_host()
    Xnp = np.arange(0, 10, dtype="u1")
    assert np.array_equal(Xh, Xnp[Xnp[2] : Xnp[5]])


def test_ctor_invalid_shape():
    with pytest.raises(TypeError):
        dpt.usm_ndarray(dict())


def test_ctor_invalid_order():
    with pytest.raises(ValueError):
        dpt.usm_ndarray((5, 5, 3), order="Z")


def test_ctor_buffer_kwarg():
    dpt.usm_ndarray(10, buffer=b"device")
    with pytest.raises(ValueError):
        dpt.usm_ndarray(10, buffer="invalid_param")
    Xusm = dpt.usm_ndarray((10, 5), dtype="c16")
    X2 = dpt.usm_ndarray(Xusm.shape, buffer=Xusm, dtype=Xusm.dtype)
    assert np.array_equal(
        Xusm.usm_data.copy_to_host(), X2.usm_data.copy_to_host()
    )
    with pytest.raises(ValueError):
        dpt.usm_ndarray(10, buffer=dict())


def test_usm_ndarray_props():
    Xusm = dpt.usm_ndarray((10, 5), dtype="c16", order="F")
    Xusm.ndim
    repr(Xusm)
    Xusm.flags
    Xusm.__sycl_usm_array_interface__
    Xusm.device
    Xusm.strides
    Xusm.real
    Xusm.imag
    try:
        dpctl.SyclQueue("cpu")
    except dpctl.SyclQueueCreationError:
        pytest.skip("Sycl device CPU was not detected")
    Xusm.to_device("cpu")


def test_datapi_device():
    X = dpt.usm_ndarray(1)
    dev_t = type(X.device)
    with pytest.raises(TypeError):
        dev_t()
    dev_t.create_device(X.device)
    dev_t.create_device(X.sycl_queue)
    dev_t.create_device(X.sycl_device)
    dev_t.create_device(X.sycl_device.filter_string)
    dev_t.create_device(None)
    X.device.sycl_context
    X.device.sycl_queue
    X.device.sycl_device
    repr(X.device)


def _pyx_capi_fnptr_to_callable(
    X, pyx_capi_name, caps_name, fn_restype=ctypes.c_void_p
):
    import sys

    mod = sys.modules[X.__class__.__module__]
    cap = mod.__pyx_capi__.get(pyx_capi_name, None)
    if cap is None:
        raise ValueError(
            "__pyx_capi__ does not export {} capsule".format(pyx_capi_name)
        )
    # construct Python callable to invoke these functions
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    fn_ptr = cap_ptr_fn(cap, caps_name)
    callable_maker_ptr = ctypes.PYFUNCTYPE(fn_restype, ctypes.py_object)
    return callable_maker_ptr(fn_ptr)


def test_pyx_capi_get_data():
    X = dpt.usm_ndarray(17)[1::2]
    get_data_fn = _pyx_capi_fnptr_to_callable(
        X,
        "usm_ndarray_get_data",
        b"char *(struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
    )
    r1 = get_data_fn(X)
    sua_iface = X.__sycl_usm_array_interface__
    assert r1 == sua_iface["data"][0] + sua_iface.get("offset") * X.itemsize


def test_pyx_capi_get_shape():
    X = dpt.usm_ndarray(17)[1::2]
    get_shape_fn = _pyx_capi_fnptr_to_callable(
        X,
        "usm_ndarray_get_shape",
        b"Py_ssize_t *(struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
    )
    c_longlong_p = ctypes.POINTER(ctypes.c_longlong)
    shape0 = ctypes.cast(get_shape_fn(X), c_longlong_p).contents.value
    assert shape0 == X.shape[0]


def test_pyx_capi_get_strides():
    X = dpt.usm_ndarray(17)[1::2]
    get_strides_fn = _pyx_capi_fnptr_to_callable(
        X,
        "usm_ndarray_get_strides",
        b"Py_ssize_t *(struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
    )
    c_longlong_p = ctypes.POINTER(ctypes.c_longlong)
    strides0_p = get_strides_fn(X)
    if strides0_p:
        strides0_p = ctypes.cast(strides0_p, c_longlong_p).contents
        strides0_p = strides0_p.value
    assert strides0_p == 0 or strides0_p == X.strides[0]


def test_pyx_capi_get_ndim():
    X = dpt.usm_ndarray(17)[1::2]
    get_ndim_fn = _pyx_capi_fnptr_to_callable(
        X,
        "usm_ndarray_get_ndim",
        b"int (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_int,
    )
    assert get_ndim_fn(X) == X.ndim


def test_pyx_capi_get_typenum():
    X = dpt.usm_ndarray(17)[1::2]
    get_typenum_fn = _pyx_capi_fnptr_to_callable(
        X,
        "usm_ndarray_get_typenum",
        b"int (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_int,
    )
    typenum = get_typenum_fn(X)
    assert type(typenum) is int
    assert typenum == X.dtype.num


def test_pyx_capi_get_flags():
    X = dpt.usm_ndarray(17)[1::2]
    get_flags_fn = _pyx_capi_fnptr_to_callable(
        X,
        "usm_ndarray_get_flags",
        b"int (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_int,
    )
    flags = get_flags_fn(X)
    assert type(flags) is int and flags == X.flags


def test_pyx_capi_get_queue_ref():
    X = dpt.usm_ndarray(17)[1::2]
    get_queue_ref_fn = _pyx_capi_fnptr_to_callable(
        X,
        "usm_ndarray_get_queue_ref",
        b"DPCTLSyclQueueRef (struct PyUSMArrayObject *)",
        fn_restype=ctypes.c_void_p,
    )
    queue_ref = get_queue_ref_fn(X)  # address of a copy, should be unequal
    assert queue_ref != X.sycl_queue.addressof_ref()


def _pyx_capi_int(X, pyx_capi_name, caps_name=b"int", val_restype=ctypes.c_int):
    import sys

    mod = sys.modules[X.__class__.__module__]
    cap = mod.__pyx_capi__.get(pyx_capi_name, None)
    if cap is None:
        raise ValueError(
            "__pyx_capi__ does not export {} capsule".format(pyx_capi_name)
        )
    # construct Python callable to invoke these functions
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    cap_ptr = cap_ptr_fn(cap, caps_name)
    val_ptr = ctypes.cast(cap_ptr, ctypes.POINTER(val_restype))
    return val_ptr.contents.value


def test_pyx_capi_check_constants():
    X = dpt.usm_ndarray(17)[1::2]
    cc_flag = _pyx_capi_int(X, "USM_ARRAY_C_CONTIGUOUS")
    assert cc_flag > 0 and 0 == (cc_flag & (cc_flag - 1))
    fc_flag = _pyx_capi_int(X, "USM_ARRAY_F_CONTIGUOUS")
    assert fc_flag > 0 and 0 == (fc_flag & (fc_flag - 1))
    w_flag = _pyx_capi_int(X, "USM_ARRAY_WRITEABLE")
    assert w_flag > 0 and 0 == (w_flag & (w_flag - 1))

    bool_typenum = _pyx_capi_int(X, "UAR_BOOL")
    assert bool_typenum == np.dtype("bool_").num

    byte_typenum = _pyx_capi_int(X, "UAR_BYTE")
    assert byte_typenum == np.dtype(np.byte).num
    ubyte_typenum = _pyx_capi_int(X, "UAR_UBYTE")
    assert ubyte_typenum == np.dtype(np.ubyte).num

    short_typenum = _pyx_capi_int(X, "UAR_SHORT")
    assert short_typenum == np.dtype(np.short).num
    ushort_typenum = _pyx_capi_int(X, "UAR_USHORT")
    assert ushort_typenum == np.dtype(np.ushort).num

    int_typenum = _pyx_capi_int(X, "UAR_INT")
    assert int_typenum == np.dtype(np.intc).num
    uint_typenum = _pyx_capi_int(X, "UAR_UINT")
    assert uint_typenum == np.dtype(np.uintc).num

    long_typenum = _pyx_capi_int(X, "UAR_LONG")
    assert long_typenum == np.dtype(np.int_).num
    ulong_typenum = _pyx_capi_int(X, "UAR_ULONG")
    assert ulong_typenum == np.dtype(np.uint).num

    longlong_typenum = _pyx_capi_int(X, "UAR_LONGLONG")
    assert longlong_typenum == np.dtype(np.longlong).num
    ulonglong_typenum = _pyx_capi_int(X, "UAR_ULONGLONG")
    assert ulonglong_typenum == np.dtype(np.ulonglong).num

    half_typenum = _pyx_capi_int(X, "UAR_HALF")
    assert half_typenum == np.dtype(np.half).num
    float_typenum = _pyx_capi_int(X, "UAR_FLOAT")
    assert float_typenum == np.dtype(np.single).num
    double_typenum = _pyx_capi_int(X, "UAR_DOUBLE")
    assert double_typenum == np.dtype(np.double).num

    cfloat_typenum = _pyx_capi_int(X, "UAR_CFLOAT")
    assert cfloat_typenum == np.dtype(np.csingle).num
    cdouble_typenum = _pyx_capi_int(X, "UAR_CDOUBLE")
    assert cdouble_typenum == np.dtype(np.cdouble).num


@pytest.mark.parametrize(
    "shape", [tuple(), (1,), (5,), (2, 3), (2, 3, 4), (2, 2, 2, 2, 2)]
)
@pytest.mark.parametrize(
    "dtype",
    [
        "b1",
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_tofrom_numpy(shape, dtype, usm_type):
    q = dpctl.SyclQueue()
    Xnp = np.zeros(shape, dtype=dtype)
    Xusm = dpt.from_numpy(Xnp, usm_type=usm_type, queue=q)
    Ynp = np.ones(shape, dtype=dtype)
    ind = (slice(None, None, None),) * Ynp.ndim
    Xusm[ind] = Ynp
    assert np.array_equal(dpt.to_numpy(Xusm), Ynp)


@pytest.mark.parametrize(
    "dtype",
    [
        "b1",
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
@pytest.mark.parametrize("src_usm_type", ["device", "shared", "host"])
@pytest.mark.parametrize("dst_usm_type", ["device", "shared", "host"])
def test_setitem_same_dtype(dtype, src_usm_type, dst_usm_type):
    Xnp = (
        np.random.randint(-10, 10, size=2 * 3 * 4)
        .astype(dtype)
        .reshape((2, 4, 3))
    )
    Znp = np.zeros(
        (
            2,
            4,
            3,
        ),
        dtype=dtype,
    )
    Zusm_0d = dpt.from_numpy(Znp[0, 0, 0], usm_type=dst_usm_type)
    ind = (-1, -1, -1)
    Xusm_0d = dpt.from_numpy(Xnp[ind], usm_type=src_usm_type)
    Zusm_0d[Ellipsis] = Xusm_0d
    assert np.array_equal(dpt.to_numpy(Zusm_0d), Xnp[ind])
    Zusm_1d = dpt.from_numpy(Znp[0, 1:3, 0], usm_type=dst_usm_type)
    ind = (-1, slice(0, 2, None), -1)
    Xusm_1d = dpt.from_numpy(Xnp[ind], usm_type=src_usm_type)
    Zusm_1d[Ellipsis] = Xusm_1d
    assert np.array_equal(dpt.to_numpy(Zusm_1d), Xnp[ind])
    Zusm_2d = dpt.from_numpy(Znp[:, 1:3, 0], usm_type=dst_usm_type)[::-1]
    Xusm_2d = dpt.from_numpy(Xnp[:, 1:4, -1], usm_type=src_usm_type)
    Zusm_2d[:] = Xusm_2d[:, 0:2]
    assert np.array_equal(dpt.to_numpy(Zusm_2d), Xnp[:, 1:3, -1])
    Zusm_3d = dpt.from_numpy(Znp, usm_type=dst_usm_type)
    Xusm_3d = dpt.from_numpy(Xnp, usm_type=src_usm_type)
    Zusm_3d[:] = Xusm_3d
    assert np.array_equal(dpt.to_numpy(Zusm_3d), Xnp)
    Zusm_3d[::-1] = Xusm_3d[::-1]
    assert np.array_equal(dpt.to_numpy(Zusm_3d), Xnp)
    Zusm_3d[:] = Xusm_3d[0]
    R1 = dpt.to_numpy(Zusm_3d)
    R2 = np.broadcast_to(Xnp[0], R1.shape)
    assert R1.shape == R2.shape
    assert np.allclose(R1, R2)


@pytest.mark.parametrize(
    "dtype",
    [
        "b1",
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_setitem_scalar(dtype, usm_type):
    X = dpt.usm_ndarray((6, 6), dtype=dtype, buffer=usm_type)
    for i in range(X.size):
        X[np.unravel_index(i, X.shape)] = np.asarray(i, dtype=dtype)
    assert np.array_equal(
        dpt.to_numpy(X), np.arange(X.size).astype(dtype).reshape(X.shape)
    )
    Y = dpt.usm_ndarray((2, 3), dtype=dtype, buffer=usm_type)
    for i in range(Y.size):
        Y[np.unravel_index(i, Y.shape)] = i
    assert np.array_equal(
        dpt.to_numpy(Y), np.arange(Y.size).astype(dtype).reshape(Y.shape)
    )


def test_setitem_errors():
    X = dpt.usm_ndarray((4,), dtype="u1")
    Y = dpt.usm_ndarray((4, 2), dtype="u1")
    with pytest.raises(ValueError):
        X[:] = Y
    with pytest.raises(ValueError):
        X[:] = Y[:, 0:1]
    X[:] = Y[None, :, 0]


def test_setitem_different_dtypes():
    X = dpt.from_numpy(np.ones(10, "f4"))
    Y = dpt.from_numpy(np.zeros(10, "f4"))
    Z = dpt.usm_ndarray((20,), "d")
    Z[::2] = X
    Z[1::2] = Y
    assert np.allclose(dpt.asnumpy(Z), np.tile(np.array([1, 0], "d"), 10))


def test_shape_setter():
    def cc_strides(sh):
        return np.empty(sh, dtype="u1").strides

    def relaxed_strides_equal(st1, st2, sh):
        eq_ = True
        for s1, s2, d in zip(st1, st2, sh):
            eq_ = eq_ and ((d == 1) or (s1 == s2))
        return eq_

    sh_s = (2 * 3 * 4 * 5,)
    sh_f = (
        2,
        3,
        4,
        5,
    )
    X = dpt.usm_ndarray(sh_s, dtype="d")
    expected_flags = X.flags
    X.shape = sh_f
    assert X.shape == sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)
    assert X.flags == expected_flags

    sh_s = (
        2,
        12,
        5,
    )
    sh_f = (
        2,
        3,
        4,
        5,
    )
    X = dpt.usm_ndarray(sh_s, dtype="d", order="C")
    X.shape = sh_f
    assert X.shape == sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)

    sh_s = (2, 3, 4, 5)
    sh_f = (4, 3, 2, 5)
    X = dpt.usm_ndarray(sh_s, dtype="d")
    X.shape = sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)

    sh_s = (2, 3, 4, 5)
    sh_f = (4, 3, 1, 2, 5)
    X = dpt.usm_ndarray(sh_s, dtype="d")
    X.shape = sh_f
    assert relaxed_strides_equal(X.strides, cc_strides(sh_f), sh_f)

    X = dpt.usm_ndarray(sh_s, dtype="d")
    with pytest.raises(TypeError):
        X.shape = "abcbe"
    X = dpt.usm_ndarray((4, 4), dtype="d")[::2, ::2]
    with pytest.raises(AttributeError):
        X.shape = (4,)


def test_len():
    X = dpt.usm_ndarray(1, "i4")
    assert len(X) == 1
    X = dpt.usm_ndarray((2, 1), "i4")
    assert len(X) == 2
    X = dpt.usm_ndarray(tuple(), "i4")
    with pytest.raises(TypeError):
        len(X)


def test_array_namespace():
    X = dpt.usm_ndarray(1, "i4")
    X.__array_namespace__()
    X._set_namespace(dpt)
    assert X.__array_namespace__() is dpt


def test_dlpack():
    X = dpt.usm_ndarray(1, "i4")
    X.__dlpack_device__()
    X.__dlpack__(stream=None)


def test_to_device():
    X = dpt.usm_ndarray(1, "d")
    for dev in dpctl.get_devices():
        if dev.default_selector_score > 0:
            Y = X.to_device(dev)
            assert Y.sycl_device == dev


def test_astype():
    X = dpt.usm_ndarray((5, 5), "i4")
    X[:] = np.full((5, 5), 7, dtype="i4")
    Y = dpt.astype(X, "c16", order="C")
    assert np.allclose(dpt.to_numpy(Y), np.full((5, 5), 7, dtype="c16"))
    Y = dpt.astype(X, "f2", order="K")
    assert np.allclose(dpt.to_numpy(Y), np.full((5, 5), 7, dtype="f2"))
    Y = dpt.astype(X, "i4", order="K", copy=False)
    assert Y.usm_data is X.usm_data


def test_astype_invalid_order():
    X = dpt.usm_ndarray(5, "i4")
    with pytest.raises(ValueError):
        dpt.astype(X, "i4", order="WRONG")


def test_copy():
    X = dpt.usm_ndarray((5, 5), "i4")[2:4, 1:4]
    X[:] = 42
    Yc = dpt.copy(X, order="C")
    Yf = dpt.copy(X, order="F")
    Ya = dpt.copy(X, order="A")
    Yk = dpt.copy(X, order="K")
    assert Yc.usm_data is not X.usm_data
    assert Yf.usm_data is not X.usm_data
    assert Ya.usm_data is not X.usm_data
    assert Yk.usm_data is not X.usm_data
    assert Yc.strides == (3, 1)
    assert Yf.strides == (1, 2)
    assert Ya.strides == (3, 1)
    assert Yk.strides == (3, 1)
    ref = np.full(X.shape, 42, dtype=X.dtype)
    assert np.array_equal(dpt.asnumpy(Yc), ref)
    assert np.array_equal(dpt.asnumpy(Yf), ref)
    assert np.array_equal(dpt.asnumpy(Ya), ref)
    assert np.array_equal(dpt.asnumpy(Yk), ref)


def test_ctor_invalid():
    m = dpm.MemoryUSMShared(12)
    with pytest.raises(ValueError):
        dpt.usm_ndarray((4,), dtype="i4", buffer=m)
    m = dpm.MemoryUSMShared(64)
    with pytest.raises(ValueError):
        dpt.usm_ndarray((4,), dtype="u1", buffer=m, strides={"not": "valid"})


def test_reshape():
    X = dpt.usm_ndarray((5, 5), "i4")
    # can be done as views
    Y = dpt.reshape(X, (25,))
    assert Y.shape == (25,)
    Z = X[::2, ::2]
    # requires a copy
    W = dpt.reshape(Z, (Z.size,), order="F")
    assert W.shape == (Z.size,)
    with pytest.raises(TypeError):
        dpt.reshape("invalid")
    with pytest.raises(ValueError):
        dpt.reshape(Z, (2, 2, 2, 2, 2))
    with pytest.raises(ValueError):
        dpt.reshape(Z, Z.shape, order="invalid")
    W = dpt.reshape(Z, (-1,), order="C")
    assert W.shape == (Z.size,)
