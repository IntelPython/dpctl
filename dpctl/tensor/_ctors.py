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

import operator

import numpy as np

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import dpctl.utils
from dpctl.tensor._device import normalize_queue_device

_empty_tuple = tuple()
_host_set = frozenset([None])


def _get_dtype(dtype, sycl_obj, ref_type=None):
    if dtype is None:
        if ref_type in [None, float] or np.issubdtype(ref_type, np.floating):
            dtype = ti.default_device_fp_type(sycl_obj)
            return dpt.dtype(dtype)
        elif ref_type in [bool, np.bool_]:
            dtype = ti.default_device_bool_type(sycl_obj)
            return dpt.dtype(dtype)
        elif ref_type is int or np.issubdtype(ref_type, np.integer):
            dtype = ti.default_device_int_type(sycl_obj)
            return dpt.dtype(dtype)
        elif ref_type is complex or np.issubdtype(ref_type, np.complexfloating):
            dtype = ti.default_device_complex_type(sycl_obj)
            return dpt.dtype(dtype)
        else:
            raise TypeError(f"Reference type {ref_type} not recognized.")
    else:
        return dpt.dtype(dtype)


def _array_info_dispatch(obj):
    if isinstance(obj, dpt.usm_ndarray):
        return obj.shape, obj.dtype, frozenset([obj.sycl_queue])
    elif isinstance(obj, np.ndarray):
        return obj.shape, obj.dtype, _host_set
    elif isinstance(obj, range):
        return (len(obj),), int, _host_set
    elif isinstance(obj, bool):
        return _empty_tuple, bool, _host_set
    elif isinstance(obj, float):
        return _empty_tuple, float, _host_set
    elif isinstance(obj, int):
        return _empty_tuple, int, _host_set
    elif isinstance(obj, complex):
        return _empty_tuple, complex, _host_set
    elif isinstance(obj, (list, tuple, range)):
        return _array_info_sequence(obj)
    elif any(
        isinstance(obj, s)
        for s in [np.integer, np.floating, np.complexfloating, np.bool_]
    ):
        return _empty_tuple, obj.dtype, _host_set
    else:
        raise ValueError(type(obj))


def _array_info_sequence(li):
    assert isinstance(li, (list, tuple, range))
    n = len(li)
    dim = None
    dt = None
    device = frozenset()
    for el in li:
        el_dim, el_dt, el_dev = _array_info_dispatch(el)
        if dim is None:
            dim = el_dim
            dt = np.promote_types(el_dt, el_dt)
            device = device.union(el_dev)
        elif el_dim == dim:
            dt = np.promote_types(dt, el_dt)
            device = device.union(el_dev)
        else:
            raise ValueError(
                "Inconsistent dimensions, {} and {}".format(dim, el_dim)
            )
    if dim is None:
        dim = tuple()
        dt = float
        device = _host_set
    return (n,) + dim, dt, device


def _asarray_from_usm_ndarray(
    usm_ndary,
    dtype=None,
    copy=None,
    usm_type=None,
    sycl_queue=None,
    order="K",
):
    if not isinstance(usm_ndary, dpt.usm_ndarray):
        raise TypeError(
            f"Expected dpctl.tensor.usm_ndarray, got {type(usm_ndary)}"
        )
    if dtype is None:
        dtype = usm_ndary.dtype
    if usm_type is None:
        usm_type = usm_ndary.usm_type
    if sycl_queue is not None:
        exec_q = dpctl.utils.get_execution_queue(
            [usm_ndary.sycl_queue, sycl_queue]
        )
        copy_q = normalize_queue_device(sycl_queue=sycl_queue, device=exec_q)
    else:
        copy_q = usm_ndary.sycl_queue
    # Conditions for zero copy:
    can_zero_copy = copy is not True
    #    dtype is unchanged
    can_zero_copy = can_zero_copy and dtype == usm_ndary.dtype
    #    USM allocation type is unchanged
    can_zero_copy = can_zero_copy and usm_type == usm_ndary.usm_type
    #    sycl_queue is unchanged
    can_zero_copy = can_zero_copy and copy_q is usm_ndary.sycl_queue
    #    order is unchanged
    c_contig = usm_ndary.flags.c_contiguous
    f_contig = usm_ndary.flags.f_contiguous
    fc_contig = usm_ndary.flags.forc
    if can_zero_copy:
        if order == "C" and c_contig:
            pass
        elif order == "F" and f_contig:
            pass
        elif order == "A" and fc_contig:
            pass
        elif order == "K":
            pass
        else:
            can_zero_copy = False
    if copy is False and can_zero_copy is False:
        raise ValueError("asarray(..., copy=False) is not possible")
    if can_zero_copy:
        return usm_ndary
    if order == "A":
        order = "F" if f_contig and not c_contig else "C"
    if order == "K" and fc_contig:
        order = "C" if c_contig else "F"
    if order == "K":
        # new USM allocation
        res = dpt.usm_ndarray(
            usm_ndary.shape,
            dtype=dtype,
            buffer=usm_type,
            order="C",
            buffer_ctor_kwargs={"queue": copy_q},
        )
        original_strides = usm_ndary.strides
        ind = sorted(
            range(usm_ndary.ndim),
            key=lambda i: abs(original_strides[i]),
            reverse=True,
        )
        new_strides = tuple(res.strides[ind[i]] for i in ind)
        # reuse previously made USM allocation
        res = dpt.usm_ndarray(
            usm_ndary.shape,
            dtype=res.dtype,
            buffer=res.usm_data,
            strides=new_strides,
        )
    else:
        res = dpt.usm_ndarray(
            usm_ndary.shape,
            dtype=dtype,
            buffer=usm_type,
            order=order,
            buffer_ctor_kwargs={"queue": copy_q},
        )
    # FIXME: call copy_to when implemented
    res[(slice(None, None, None),) * res.ndim] = usm_ndary
    return res


def _asarray_from_numpy_ndarray(
    ary, dtype=None, usm_type=None, sycl_queue=None, order="K"
):
    if not isinstance(ary, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(ary)}")
    if usm_type is None:
        usm_type = "device"
    copy_q = normalize_queue_device(sycl_queue=None, device=sycl_queue)
    if ary.dtype.char not in "?bBhHiIlLqQefdFD":
        raise TypeError(
            f"Numpy array of data type {ary.dtype} is not supported. "
            "Please convert the input to an array with numeric data type."
        )
    if dtype is None:
        ary_dtype = ary.dtype
        dtype = _get_dtype(dtype, copy_q, ref_type=ary_dtype)
        if dtype.itemsize > ary_dtype.itemsize:
            dtype = ary_dtype
    f_contig = ary.flags["F"]
    c_contig = ary.flags["C"]
    fc_contig = f_contig or c_contig
    if order == "A":
        order = "F" if f_contig and not c_contig else "C"
    if order == "K" and fc_contig:
        order = "C" if c_contig else "F"
    if order == "K":
        # new USM allocation
        res = dpt.usm_ndarray(
            ary.shape,
            dtype=dtype,
            buffer=usm_type,
            order="C",
            buffer_ctor_kwargs={"queue": copy_q},
        )
        original_strides = ary.strides
        ind = sorted(
            range(ary.ndim),
            key=lambda i: abs(original_strides[i]),
            reverse=True,
        )
        new_strides = tuple(res.strides[ind[i]] for i in ind)
        # reuse previously made USM allocation
        res = dpt.usm_ndarray(
            res.shape, dtype=res.dtype, buffer=res.usm_data, strides=new_strides
        )
    else:
        res = dpt.usm_ndarray(
            ary.shape,
            dtype=dtype,
            buffer=usm_type,
            order=order,
            buffer_ctor_kwargs={"queue": copy_q},
        )
    # FIXME: call copy_to when implemented
    res[(slice(None, None, None),) * res.ndim] = ary
    return res


def _is_object_with_buffer_protocol(obj):
    "Returns True if object support Python buffer protocol"
    try:
        # use context manager to ensure
        # buffer is instantly released
        with memoryview(obj):
            return True
    except TypeError:
        return False


def asarray(
    obj,
    dtype=None,
    device=None,
    copy=None,
    usm_type=None,
    sycl_queue=None,
    order="K",
):
    """
    Converts `obj` to :class:`dpctl.tensor.usm_ndarray`.

    Args:
        obj: Python object to convert. Can be an instance of `usm_ndarray`,
            an object representing SYCL USM allocation and implementing
            `__sycl_usm_array_interface__` protocol, an instance
            of `numpy.ndarray`, an object supporting Python buffer protocol,
            a Python scalar, or a (possibly nested) sequence of Python scalars.
        dtype (data type, optional): output array data type. If `dtype` is
            `None`, the output array data type is inferred from data types in
            `obj`. Default: `None`
        copy (`bool`, optional): boolean indicating whether or not to copy the
            input. If `True`, always creates a copy. If `False`, need to copy
            raises `ValueError`. If `None`, try to reuse existing memory
            allocations if possible, but allowed to perform a copy otherwise.
            Default: `None`.
        order ("C","F","A","K", optional): memory layout of the output array.
            Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. For `usm_type=None` the allocation
            type is inferred from the input if `obj` has USM allocation, or
            `"device"` is used instead. Default: `None`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    # 1. Check that copy is a valid keyword
    if copy not in [None, True, False]:
        raise TypeError(
            "Recognized copy keyword values should be True, False, or None"
        )
    # 2. Check that dtype is None, or a valid dtype
    if dtype is not None:
        dtype = dpt.dtype(dtype)
    # 3. Validate order
    if not isinstance(order, str):
        raise TypeError(
            f"Expected order keyword to be of type str, got {type(order)}"
        )
    if len(order) == 0 or order[0] not in "KkAaCcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'K', 'A', 'F', or 'C'."
        )
    else:
        order = order[0].upper()
    # 4. Check that usm_type is None, or a valid value
    dpctl.utils.validate_usm_type(usm_type, allow_none=True)
    # 5. Normalize device/sycl_queue [keep it None if was None]
    if device is not None or sycl_queue is not None:
        sycl_queue = normalize_queue_device(
            sycl_queue=sycl_queue, device=device
        )

    # handle instance(obj, usm_ndarray)
    if isinstance(obj, dpt.usm_ndarray):
        return _asarray_from_usm_ndarray(
            obj,
            dtype=dtype,
            copy=copy,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
            order=order,
        )
    elif hasattr(obj, "__sycl_usm_array_interface__"):
        sua_iface = getattr(obj, "__sycl_usm_array_interface__")
        membuf = dpm.as_usm_memory(obj)
        ary = dpt.usm_ndarray(
            sua_iface["shape"],
            dtype=sua_iface["typestr"],
            buffer=membuf,
            strides=sua_iface.get("strides", None),
        )
        return _asarray_from_usm_ndarray(
            ary,
            dtype=dtype,
            copy=copy,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
            order=order,
        )
    elif isinstance(obj, np.ndarray):
        if copy is False:
            raise ValueError(
                "Converting numpy.ndarray to usm_ndarray requires a copy"
            )
        return _asarray_from_numpy_ndarray(
            obj,
            dtype=dtype,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
            order=order,
        )
    elif _is_object_with_buffer_protocol(obj):
        if copy is False:
            raise ValueError(
                f"Converting {type(obj)} to usm_ndarray requires a copy"
            )
        return _asarray_from_numpy_ndarray(
            np.array(obj),
            dtype=dtype,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
            order=order,
        )
    elif isinstance(obj, (list, tuple, range)):
        if copy is False:
            raise ValueError(
                "Converting Python sequence to usm_ndarray requires a copy"
            )
        _, dt, devs = _array_info_sequence(obj)
        if devs == _host_set:
            return _asarray_from_numpy_ndarray(
                np.asarray(obj, dtype=dtype, order=order),
                dtype=dtype,
                usm_type=usm_type,
                sycl_queue=sycl_queue,
                order=order,
            )
        # for sequences
        raise NotImplementedError(
            "Converting Python sequences is not implemented"
        )
    if copy is False:
        raise ValueError(
            f"Converting {type(obj)} to usm_ndarray requires a copy"
        )
    # obj is a scalar, create 0d array
    return _asarray_from_numpy_ndarray(
        np.asarray(obj),
        dtype=dtype,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        order="C",
    )


def empty(
    sh, dtype=None, order="C", device=None, usm_type="device", sycl_queue=None
):
    """
    Creates `usm_ndarray` from uninitialized USM allocation.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    res = dpt.usm_ndarray(
        sh,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    return res


def _coerce_and_infer_dt(*args, dt, sycl_queue, err_msg, allow_bool=False):
    "Deduce arange type from sequence spec"
    nd, seq_dt, d = _array_info_sequence(args)
    if d != _host_set or nd != (len(args),):
        raise ValueError(err_msg)
    dt = _get_dtype(dt, sycl_queue, ref_type=seq_dt)
    if np.issubdtype(dt, np.integer):
        return tuple(int(v) for v in args), dt
    elif np.issubdtype(dt, np.floating):
        return tuple(float(v) for v in args), dt
    elif np.issubdtype(dt, np.complexfloating):
        return tuple(complex(v) for v in args), dt
    elif allow_bool and dt.char == "?":
        return tuple(bool(v) for v in args), dt
    else:
        raise ValueError(f"Data type {dt} is not supported")


def _round_for_arange(tmp):
    k = int(tmp)
    if k > 0 and float(k) < tmp:
        tmp = tmp + 1
    return tmp


def _get_arange_length(start, stop, step):
    "Compute length of arange sequence"
    span = stop - start
    if type(step) in [int, float] and type(span) in [int, float]:
        return _round_for_arange(span / step)
    tmp = span / step
    if type(tmp) is complex and tmp.imag == 0:
        tmp = tmp.real
    else:
        return tmp
    return _round_for_arange(tmp)


def arange(
    start,
    /,
    stop=None,
    step=1,
    *,
    dtype=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """ arange(start, /, stop=None, step=1, *, dtype=None, \
               device=None, usm_type="device", sycl_queue=None) -> usm_ndarray

    Args:
        start:
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `'device'`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if stop is None:
        stop = start
        start = 0
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    (start, stop, step,), dt = _coerce_and_infer_dt(
        start,
        stop,
        step,
        dt=dtype,
        sycl_queue=sycl_queue,
        err_msg="start, stop, and step must be Python scalars",
        allow_bool=False,
    )
    try:
        tmp = _get_arange_length(start, stop, step)
        sh = int(tmp)
        if sh < 0:
            sh = 0
    except TypeError:
        sh = 0
    res = dpt.usm_ndarray(
        (sh,),
        dtype=dt,
        buffer=usm_type,
        order="C",
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    _step = (start + step) - start
    _step = dt.type(_step)
    _start = dt.type(start)
    hev, _ = ti._linspace_step(_start, _step, res, sycl_queue)
    hev.wait()
    return res


def zeros(
    sh, dtype=None, order="C", device=None, usm_type="device", sycl_queue=None
):
    """
    Creates `usm_ndarray` with zero elements.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    res = dpt.usm_ndarray(
        sh,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    res.usm_data.memset()
    return res


def ones(
    sh, dtype=None, order="C", device=None, usm_type="device", sycl_queue=None
):
    """
    Creates `usm_ndarray` with elements of one.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    res = dpt.usm_ndarray(
        sh,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    hev, ev = ti._full_usm_ndarray(1, res, sycl_queue)
    hev.wait()
    return res


def full(
    sh,
    fill_value,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Creates `usm_ndarray` with elements of one.

    Args:
        shape (tuple): Dimensions of the array to be created.
        fill_value (int,float,complex): fill value
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue, ref_type=type(fill_value))
    res = dpt.usm_ndarray(
        sh,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    hev, ev = ti._full_usm_ndarray(fill_value, res, sycl_queue)
    hev.wait()
    return res


def empty_like(
    x, dtype=None, order="C", device=None, usm_type=None, sycl_queue=None
):
    """
    Creates `usm_ndarray` from uninitialized USM allocation.

    Args:
        x (usm_ndarray): Input array from which to derive the output array
            shape.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    if dtype is None:
        dtype = x.dtype
    if usm_type is None:
        usm_type = x.usm_type
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if device is None and sycl_queue is None:
        device = x.device
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    sh = x.shape
    dtype = dpt.dtype(dtype)
    res = dpt.usm_ndarray(
        sh,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    return res


def zeros_like(
    x, dtype=None, order="C", device=None, usm_type=None, sycl_queue=None
):
    """
    Creates `usm_ndarray` from USM allocation initialized with zeros.

    Args:
        x (usm_ndarray): Input array from which to derive the output array
            shape.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    if dtype is None:
        dtype = x.dtype
    if usm_type is None:
        usm_type = x.usm_type
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if device is None and sycl_queue is None:
        device = x.device
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    sh = x.shape
    dtype = dpt.dtype(dtype)
    return zeros(
        sh,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def ones_like(
    x, dtype=None, order="C", device=None, usm_type=None, sycl_queue=None
):
    """
    Creates `usm_ndarray` from USM allocation initialized with zeros.

    Args:
        x (usm_ndarray): Input array from which to derive the output array
            shape.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    if dtype is None:
        dtype = x.dtype
    if usm_type is None:
        usm_type = x.usm_type
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if device is None and sycl_queue is None:
        device = x.device
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    sh = x.shape
    dtype = dpt.dtype(dtype)
    return ones(
        sh,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def full_like(
    x,
    fill_value,
    dtype=None,
    order="C",
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Creates `usm_ndarray` from USM allocation initialized with zeros.

    Args:
        x (usm_ndarray): Input array from which to derive the output array
            shape.
        fill_value: the value to fill array with
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    if dtype is None:
        dtype = x.dtype
    if usm_type is None:
        usm_type = x.usm_type
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if device is None and sycl_queue is None:
        device = x.device
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    sh = x.shape
    dtype = dpt.dtype(dtype)
    return full(
        sh,
        fill_value,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def linspace(
    start,
    stop,
    /,
    num,
    *,
    dtype=None,
    device=None,
    endpoint=True,
    sycl_queue=None,
    usm_type="device",
):
    """
    linspace(start, stop, num, dtype=None, device=None, endpoint=True,
        sycl_queue=None, usm_type=None): usm_ndarray

    Returns evenly spaced numbers of specified interval.

    Args:
        start: the start of the interval.
        stop: the end of the interval. If the `endpoint` is `False`, the
            function must generate `num+1` evenly spaced points starting
            with `start` and ending with `stop` and exclude the `stop`
            from the returned array such that the returned array consists
            of evenly spaced numbers over the half-open interval
            `[start, stop)`. If `endpoint` is `True`, the output
            array must consist of evenly spaced numbers over the closed
            interval `[start, stop]`. Default: `True`.
        num: number of samples. Must be a non-negative integer; otherwise,
            the function must raise an exception.
        dtype: output array data type. Should be a floating data type.
            If `dtype` is `None`, the output array must be the default
            floating point data type. Default: `None`.
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
        endpoint: boolean indicating whether to include `stop` in the
            interval. Default: `True`.
    """
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if endpoint not in [True, False]:
        raise TypeError("endpoint keyword argument must be of boolean type")
    num = operator.index(num)
    if num < 0:
        raise ValueError("Number of points must be non-negative")
    ((start, stop,), dt) = _coerce_and_infer_dt(
        start,
        stop,
        dt=dtype,
        sycl_queue=sycl_queue,
        err_msg="start and stop must be Python scalars.",
        allow_bool=True,
    )
    if dtype is None and np.issubdtype(dt, np.integer):
        dt = ti.default_device_fp_type(sycl_queue)
        dt = dpt.dtype(dt)
        start = float(start)
        stop = float(stop)
    res = dpt.empty(num, dtype=dt, sycl_queue=sycl_queue)
    hev, _ = ti._linspace_affine(
        start, stop, dst=res, include_endpoint=endpoint, sycl_queue=sycl_queue
    )
    hev.wait()
    return res


def eye(
    n_rows,
    n_cols=None,
    /,
    *,
    k=0,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    eye(n_rows, n_cols = None, /, *, k = 0, dtype = None, \
    device = None, usm_type="device", sycl_queue=None) -> usm_ndarray

    Creates `usm_ndarray` with ones on the `k`th diagonal.

    Args:
        n_rows: number of rows in the output array.
        n_cols (optional): number of columns in the output array. If None,
            n_cols = n_rows. Default: `None`.
        k: index of the diagonal, with 0 as the main diagonal.
            A positive value of k is a superdiagonal, a negative value
            is a subdiagonal.
            Raises `TypeError` if k is not an integer.
            Default: `0`.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: None
        order ("C" or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    n_rows = operator.index(n_rows)
    n_cols = n_rows if n_cols is None else operator.index(n_cols)
    k = operator.index(k)
    if k >= n_cols or -k >= n_rows:
        return dpt.zeros(
            (n_rows, n_cols),
            dtype=dtype,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    res = dpt.usm_ndarray(
        (n_rows, n_cols),
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    if n_rows != 0 and n_cols != 0:
        hev, _ = ti._eye(k, dst=res, sycl_queue=sycl_queue)
        hev.wait()
    return res


def tril(X, k=0):
    """
    tril(X: usm_ndarray, k: int) -> usm_ndarray

    Returns the lower triangular part of a matrix (or a stack of matrices) X.
    """
    if type(X) is not dpt.usm_ndarray:
        raise TypeError

    k = operator.index(k)

    # F_CONTIGUOUS = 2
    order = "F" if (X.flags.f_contiguous) else "C"

    shape = X.shape
    nd = X.ndim
    if nd < 2:
        raise ValueError("Array dimensions less than 2.")

    if k >= shape[nd - 1] - 1:
        res = dpt.empty(
            X.shape, dtype=X.dtype, order=order, sycl_queue=X.sycl_queue
        )
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=X, dst=res, sycl_queue=X.sycl_queue
        )
        hev.wait()
    elif k < -shape[nd - 2]:
        res = dpt.zeros(
            X.shape, dtype=X.dtype, order=order, sycl_queue=X.sycl_queue
        )
    else:
        res = dpt.empty(
            X.shape, dtype=X.dtype, order=order, sycl_queue=X.sycl_queue
        )
        hev, _ = ti._tril(src=X, dst=res, k=k, sycl_queue=X.sycl_queue)
        hev.wait()

    return res


def triu(X, k=0):
    """
    triu(X: usm_ndarray, k: int) -> usm_ndarray

    Returns the upper triangular part of a matrix (or a stack of matrices) X.
    """
    if type(X) is not dpt.usm_ndarray:
        raise TypeError

    k = operator.index(k)

    # F_CONTIGUOUS = 2
    order = "F" if (X.flags.f_contiguous) else "C"

    shape = X.shape
    nd = X.ndim
    if nd < 2:
        raise ValueError("Array dimensions less than 2.")

    if k > shape[nd - 1]:
        res = dpt.zeros(
            X.shape, dtype=X.dtype, order=order, sycl_queue=X.sycl_queue
        )
    elif k <= -shape[nd - 2] + 1:
        res = dpt.empty(
            X.shape, dtype=X.dtype, order=order, sycl_queue=X.sycl_queue
        )
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=X, dst=res, sycl_queue=X.sycl_queue
        )
        hev.wait()
    else:
        res = dpt.empty(
            X.shape, dtype=X.dtype, order=order, sycl_queue=X.sycl_queue
        )
        hev, _ = ti._triu(src=X, dst=res, k=k, sycl_queue=X.sycl_queue)
        hev.wait()

    return res


def meshgrid(*arrays, indexing="xy"):

    """
    meshgrid(*arrays, indexing="xy") -> list[usm_ndarray]

    Creates list of `usm_ndarray` coordinate matrices from vectors.

    Args:
        arrays: arbitrary number of one-dimensional `USM_ndarray` objects.
            If vectors are not of the same data type,
            or are not one-dimensional, raises `ValueError.`
        indexing: Cartesian (`xy`) or matrix (`ij`) indexing of output.
            For a set of `n` vectors with lengths N0, N1, N2, ...
            Cartesian indexing results in arrays of shape
            (N1, N0, N2, ...)
            matrix indexing results in arrays of shape
            (n0, N1, N2, ...)
            Default: `xy`.
    """
    ref_dt = None
    ref_unset = True
    for array in arrays:
        if not isinstance(array, dpt.usm_ndarray):
            raise TypeError(
                f"Expected instance of dpt.usm_ndarray, got {type(array)}."
            )
        if array.ndim != 1:
            raise ValueError("All arrays must be one-dimensional.")
        if ref_unset:
            ref_unset = False
            ref_dt = array.dtype
        else:
            if not ref_dt == array.dtype:
                raise ValueError(
                    "All arrays must be of the same numeric data type."
                )
    if indexing not in ["xy", "ij"]:
        raise ValueError(
            "Unrecognized indexing keyword value, expecting 'xy' or 'ij.'"
        )
    n = len(arrays)
    sh = (-1,) + (1,) * (n - 1)

    res = []
    if n > 1 and indexing == "xy":
        res.append(dpt.reshape(arrays[0], (1, -1) + sh[2:], copy=True))
        res.append(dpt.reshape(arrays[1], sh, copy=True))
        arrays, sh = arrays[2:], sh[-2:] + sh[:-2]

    for array in arrays:
        res.append(dpt.reshape(array, sh, copy=True))
        sh = sh[-1:] + sh[:-1]

    output = dpt.broadcast_arrays(*res)

    return output
