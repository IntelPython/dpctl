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


def test_properties():
    """
    Test that properties execute
    """
    X = dpt.usm_ndarray((3, 4, 5), dtype="c16")
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


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("index_dtype", ["|i8"])
def test_usm_ndarray_as_index(shape, index_dtype):
    X = dpt.usm_ndarray(shape, dtype=index_dtype)
    Xnp = np.arange(1, X.size + 1, dtype=index_dtype).reshape(shape)
    X.usm_data.copy_from_host(Xnp.reshape(-1).view("|u1"))
    Y = np.arange(X.size + 1)
    assert Y[X] == Y[1]


@pytest.mark.parametrize("shape", [(2,), (1, 2), (3, 4, 5), (0,)])
@pytest.mark.parametrize("index_dtype", ["|i8"])
def test_usm_ndarray_as_index_invalid_shape(shape, index_dtype):
    X = dpt.usm_ndarray(shape, dtype=index_dtype)
    Y = np.arange(X.size + 1)
    with pytest.raises(IndexError):
        Y[X]


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("index_dtype", ["|f8"])
def test_usm_ndarray_as_index_invalid_dtype(shape, index_dtype):
    X = dpt.usm_ndarray(shape, dtype=index_dtype)
    Y = np.arange(X.size + 1)
    with pytest.raises(IndexError):
        Y[X]


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


def test_pyx_capi():
    import ctypes
    import sys

    X = dpt.usm_ndarray(17)[1::2]
    mod = sys.modules[X.__class__.__module__]
    # get capsule storign get_context_ref function ptr
    arr_data_fn_cap = mod.__pyx_capi__["usm_ndarray_get_data"]
    arr_ndim_fn_cap = mod.__pyx_capi__["usm_ndarray_get_ndim"]
    arr_shape_fn_cap = mod.__pyx_capi__["usm_ndarray_get_shape"]
    arr_strides_fn_cap = mod.__pyx_capi__["usm_ndarray_get_strides"]
    arr_typenum_fn_cap = mod.__pyx_capi__["usm_ndarray_get_typenum"]
    arr_flags_fn_cap = mod.__pyx_capi__["usm_ndarray_get_flags"]
    arr_queue_ref_fn_cap = mod.__pyx_capi__["usm_ndarray_get_queue_ref"]
    # construct Python callable to invoke these functions
    cap_ptr_fn = ctypes.pythonapi.PyCapsule_GetPointer
    cap_ptr_fn.restype = ctypes.c_void_p
    cap_ptr_fn.argtypes = [ctypes.py_object, ctypes.c_char_p]
    callable_maker_ptr = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)
    callable_maker_int = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.py_object)
    arr_data_fn_ptr = cap_ptr_fn(
        arr_data_fn_cap, b"char *(struct PyUSMArrayObject *)"
    )
    get_data_fn = callable_maker_ptr(arr_data_fn_ptr)

    arr_ndim_fn_ptr = cap_ptr_fn(
        arr_ndim_fn_cap, b"int (struct PyUSMArrayObject *)"
    )
    get_ndim_fn = callable_maker_int(arr_ndim_fn_ptr)

    arr_shape_fn_ptr = cap_ptr_fn(
        arr_shape_fn_cap, b"Py_ssize_t *(struct PyUSMArrayObject *)"
    )
    get_shape_fn = callable_maker_ptr(arr_shape_fn_ptr)

    arr_strides_fn_ptr = cap_ptr_fn(
        arr_strides_fn_cap, b"Py_ssize_t *(struct PyUSMArrayObject *)"
    )
    get_strides_fn = callable_maker_ptr(arr_strides_fn_ptr)
    arr_typenum_fn_ptr = cap_ptr_fn(
        arr_typenum_fn_cap, b"int (struct PyUSMArrayObject *)"
    )
    get_typenum_fn = callable_maker_int(arr_typenum_fn_ptr)
    arr_flags_fn_ptr = cap_ptr_fn(
        arr_flags_fn_cap, b"int (struct PyUSMArrayObject *)"
    )
    get_flags_fn = callable_maker_int(arr_flags_fn_ptr)
    arr_queue_ref_fn_ptr = cap_ptr_fn(
        arr_queue_ref_fn_cap, b"DPCTLSyclQueueRef (struct PyUSMArrayObject *)"
    )
    get_queue_ref_fn = callable_maker_ptr(arr_queue_ref_fn_ptr)

    r1 = get_data_fn(X)
    sua_iface = X.__sycl_usm_array_interface__
    assert r1 == sua_iface["data"][0] + sua_iface.get("offset") * X.itemsize
    assert get_ndim_fn(X) == X.ndim
    c_longlong_p = ctypes.POINTER(ctypes.c_longlong)
    shape0 = ctypes.cast(get_shape_fn(X), c_longlong_p).contents.value
    assert shape0 == X.shape[0]
    strides0_p = get_strides_fn(X)
    if strides0_p:
        strides0_p = ctypes.cast(strides0_p, c_longlong_p).contents
        strides0_p = strides0_p.value
    assert strides0_p == 0 or strides0_p == X.strides[0]
    typenum = get_typenum_fn(X)
    assert type(typenum) is int
    flags = get_flags_fn(X)
    assert type(flags) is int and flags == X.flags
    queue_ref = get_queue_ref_fn(X)  # address of a copy, should be unequal
    assert queue_ref != X.sycl_queue.addressof_ref()
