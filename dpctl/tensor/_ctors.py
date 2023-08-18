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
from dpctl.tensor._copy_utils import _empty_like_orderK
from dpctl.tensor._data_types import _get_dtype
from dpctl.tensor._device import normalize_queue_device
from dpctl.tensor._usmarray import _is_object_with_buffer_protocol

__doc__ = "Implementation of creation functions in :module:`dpctl.tensor`"

_empty_tuple = tuple()
_host_set = frozenset([None])


def _array_info_dispatch(obj):
    if isinstance(obj, dpt.usm_ndarray):
        return obj.shape, obj.dtype, frozenset([obj.sycl_queue])
    if isinstance(obj, np.ndarray):
        return obj.shape, obj.dtype, _host_set
    if isinstance(obj, range):
        return (len(obj),), int, _host_set
    if isinstance(obj, bool):
        return _empty_tuple, bool, _host_set
    if isinstance(obj, float):
        return _empty_tuple, float, _host_set
    if isinstance(obj, int):
        return _empty_tuple, int, _host_set
    if isinstance(obj, complex):
        return _empty_tuple, complex, _host_set
    if isinstance(
        obj,
        (
            list,
            tuple,
        ),
    ):
        return _array_info_sequence(obj)
    if _is_object_with_buffer_protocol(obj):
        np_obj = np.array(obj)
        return np_obj.shape, np_obj.dtype, _host_set
    if hasattr(obj, "__sycl_usm_array_interface__"):
        usm_ar = _usm_ndarray_from_suai(obj)
        return usm_ar.shape, usm_ar.dtype, frozenset([usm_ar.sycl_queue])
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
            raise ValueError(f"Inconsistent dimensions, {dim} and {el_dim}")
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
        _ensure_native_dtype_device_support(dtype, copy_q.sycl_device)
        res = _empty_like_orderK(usm_ndary, dtype, usm_type, copy_q)
    else:
        _ensure_native_dtype_device_support(dtype, copy_q.sycl_device)
        res = dpt.usm_ndarray(
            usm_ndary.shape,
            dtype=dtype,
            buffer=usm_type,
            order=order,
            buffer_ctor_kwargs={"queue": copy_q},
        )
    eq = dpctl.utils.get_execution_queue([usm_ndary.sycl_queue, copy_q])
    if eq is not None:
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=usm_ndary, dst=res, sycl_queue=eq
        )
        hev.wait()
    else:
        tmp = dpt.asnumpy(usm_ndary)
        res[...] = tmp
    return res


def _map_to_device_dtype(dt, q):
    dtc = dt.char
    if dtc == "?" or np.issubdtype(dt, np.integer):
        return dt
    d = q.sycl_device
    if np.issubdtype(dt, np.floating):
        if dtc == "f":
            return dt
        if dtc == "d" and d.has_aspect_fp64:
            return dt
        if dtc == "e" and d.has_aspect_fp16:
            return dt
        return dpt.dtype("f4")
    if np.issubdtype(dt, np.complexfloating):
        if dtc == "F":
            return dt
        if dtc == "D" and d.has_aspect_fp64:
            return dt
        return dpt.dtype("c8")
    raise RuntimeError(f"Unrecognized data type '{dt}' encountered.")


def _usm_ndarray_from_suai(obj):
    sua_iface = getattr(obj, "__sycl_usm_array_interface__")
    membuf = dpm.as_usm_memory(obj)
    ary = dpt.usm_ndarray(
        sua_iface["shape"],
        dtype=sua_iface["typestr"],
        buffer=membuf,
        strides=sua_iface.get("strides", None),
    )
    return ary


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
        # deduce device-representable output data type
        dtype = _map_to_device_dtype(ary.dtype, copy_q)
    f_contig = ary.flags["F"]
    c_contig = ary.flags["C"]
    fc_contig = f_contig or c_contig
    if order == "A":
        order = "F" if f_contig and not c_contig else "C"
    if order == "K" and fc_contig:
        order = "C" if c_contig else "F"
    if order == "K":
        # new USM allocation
        _ensure_native_dtype_device_support(dtype, copy_q.sycl_device)
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
        _ensure_native_dtype_device_support(dtype, copy_q.sycl_device)
        res = dpt.usm_ndarray(
            ary.shape,
            dtype=dtype,
            buffer=usm_type,
            order=order,
            buffer_ctor_kwargs={"queue": copy_q},
        )
    res[...] = ary
    return res


def _ensure_native_dtype_device_support(dtype, dev) -> None:
    """Check that dtype is natively supported by device.

    Arg:
       dtype:
           Elemental data-type
       dev (:class:`dpctl.SyclDevice`):
           The device about which the query is being made.
    Returns:
       None
    Raise:
       ValueError - if device does not natively support this dtype.
    """
    if dtype in [dpt.float64, dpt.complex128] and not dev.has_aspect_fp64:
        raise ValueError(
            f"Device {dev.name} does not provide native support "
            "for double-precision floating point type."
        )
    if (
        dtype
        in [
            dpt.float16,
        ]
        and not dev.has_aspect_fp16
    ):
        raise ValueError(
            f"Device {dev.name} does not provide native support "
            "for half-precision floating point type."
        )


def _usm_types_walker(o, usm_types_list):
    if isinstance(o, dpt.usm_ndarray):
        usm_types_list.append(o.usm_type)
        return
    if hasattr(o, "__sycl_usm_array_interface__"):
        usm_ar = _usm_ndarray_from_suai(o)
        usm_types_list.append(usm_ar.usm_type)
        return
    if _is_object_with_buffer_protocol(o):
        return
    if isinstance(o, (int, bool, float, complex)):
        return
    if isinstance(o, (list, tuple, range)):
        for el in o:
            _usm_types_walker(el, usm_types_list)
        return
    raise TypeError


def _device_copy_walker(seq_o, res, events):
    if isinstance(seq_o, dpt.usm_ndarray):
        exec_q = res.sycl_queue
        ht_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=seq_o, dst=res, sycl_queue=exec_q
        )
        events.append(ht_ev)
        return
    if hasattr(seq_o, "__sycl_usm_array_interface__"):
        usm_ar = _usm_ndarray_from_suai(seq_o)
        exec_q = res.sycl_queue
        ht_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=usm_ar, dst=res, sycl_queue=exec_q
        )
        events.append(ht_ev)
        return
    if isinstance(seq_o, (list, tuple)):
        for i, el in enumerate(seq_o):
            _device_copy_walker(el, res[i], events)
        return
    raise TypeError


def _copy_through_host_walker(seq_o, usm_res):
    if isinstance(seq_o, dpt.usm_ndarray):
        if (
            dpctl.utils.get_execution_queue(
                (
                    usm_res.sycl_queue,
                    seq_o.sycl_queue,
                )
            )
            is None
        ):
            usm_res[...] = dpt.asnumpy(seq_o).copy()
            return
        else:
            usm_res[...] = seq_o
    if hasattr(seq_o, "__sycl_usm_array_interface__"):
        usm_ar = _usm_ndarray_from_suai(seq_o)
        if (
            dpctl.utils.get_execution_queue(
                (
                    usm_res.sycl_queue,
                    usm_ar.sycl_queue,
                )
            )
            is None
        ):
            usm_res[...] = dpt.asnumpy(usm_ar).copy()
        else:
            usm_res[...] = usm_ar
        return
    if _is_object_with_buffer_protocol(seq_o):
        np_ar = np.asarray(seq_o)
        usm_res[...] = np_ar
        return
    if isinstance(seq_o, (list, tuple)):
        for i, el in enumerate(seq_o):
            _copy_through_host_walker(el, usm_res[i])
        return
    usm_res[...] = np.asarray(seq_o)


def _asarray_from_seq(
    seq_obj,
    seq_shape,
    seq_dt,
    alloc_q,
    exec_q,
    dtype=None,
    usm_type=None,
    order="C",
):
    "`obj` is a sequence"
    if usm_type is None:
        usm_types_in_seq = []
        _usm_types_walker(seq_obj, usm_types_in_seq)
        usm_type = dpctl.utils.get_coerced_usm_type(usm_types_in_seq)
    dpctl.utils.validate_usm_type(usm_type)
    if dtype is None:
        dtype = _map_to_device_dtype(seq_dt, alloc_q)
    else:
        _mapped_dt = _map_to_device_dtype(dtype, alloc_q)
        if _mapped_dt != dtype:
            raise ValueError(
                f"Device {alloc_q.sycl_device} "
                f"does not support {dtype} natively."
            )
        dtype = _mapped_dt
    if order in "KA":
        order = "C"
    if isinstance(exec_q, dpctl.SyclQueue):
        res = dpt.empty(
            seq_shape,
            dtype=dtype,
            usm_type=usm_type,
            sycl_queue=alloc_q,
            order=order,
        )
        ht_events = []
        _device_copy_walker(seq_obj, res, ht_events)
        dpctl.SyclEvent.wait_for(ht_events)
        return res
    else:
        res = dpt.empty(
            seq_shape,
            dtype=dtype,
            usm_type=usm_type,
            sycl_queue=alloc_q,
            order=order,
        )
        _copy_through_host_walker(seq_obj, res)
        return res


def _asarray_from_seq_single_device(
    obj,
    seq_shape,
    seq_dt,
    seq_dev,
    dtype=None,
    usm_type=None,
    sycl_queue=None,
    order="C",
):
    if sycl_queue is None:
        exec_q = seq_dev
        alloc_q = seq_dev
    else:
        exec_q = dpctl.utils.get_execution_queue(
            (
                sycl_queue,
                seq_dev,
            )
        )
        alloc_q = sycl_queue
    return _asarray_from_seq(
        obj,
        seq_shape,
        seq_dt,
        alloc_q,
        exec_q,
        dtype=dtype,
        usm_type=usm_type,
        order=order,
    )


def asarray(
    obj,
    dtype=None,
    device=None,
    copy=None,
    usm_type=None,
    sycl_queue=None,
    order="K",
):
    """ asarray(obj, dtype=None, copy=None, device=None, \
           usm_type=None, sycl_queue=None, order="K")

    Converts `obj` to :class:`dpctl.tensor.usm_ndarray`.

    Args:
        obj: Python object to convert. Can be an instance of
            :class:`dpctl.tensor.usm_ndarray`,
            an object representing SYCL USM allocation and implementing
            `__sycl_usm_array_interface__` protocol, an instance
            of :class:`numpy.ndarray`, an object supporting Python buffer
            protocol, a Python scalar, or a (possibly nested) sequence of
            Python scalars.
        dtype (data type, optional): output array data type. If `dtype` is
            `None`, the output array data type is inferred from data types in
            `obj`. Default: `None`.
        copy (`bool`, optional): boolean indicating whether or not to copy the
            input. If `True`, always creates a copy. If `False`, need to copy
            raises `ValueError`. If `None`, try to reuse existing memory
            allocations if possible, but allowed to perform a copy otherwise.
            Default: `None`.
        order ("C","F","A","K", optional): memory layout of the output array.
            Default: "K"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. For `usm_type=None` the allocation
            type is inferred from the input if `obj` has USM allocation, or
            `"device"` is used instead. Default: `None`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation and
            copying. Default: `None`.

    Returns:
        usm_ndarray:
            Array created from input object.
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
    if hasattr(obj, "__sycl_usm_array_interface__"):
        ary = _usm_ndarray_from_suai(obj)
        return _asarray_from_usm_ndarray(
            ary,
            dtype=dtype,
            copy=copy,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
            order=order,
        )
    if isinstance(obj, np.ndarray):
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
    if _is_object_with_buffer_protocol(obj):
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
    if isinstance(obj, (list, tuple, range)):
        if copy is False:
            raise ValueError(
                "Converting Python sequence to usm_ndarray requires a copy"
            )
        seq_shape, seq_dt, devs = _array_info_sequence(obj)
        if devs == _host_set:
            return _asarray_from_numpy_ndarray(
                np.asarray(obj, dtype=dtype, order=order),
                dtype=dtype,
                usm_type=usm_type,
                sycl_queue=sycl_queue,
                order=order,
            )
        elif len(devs) == 1:
            seq_dev = list(devs)[0]
            return _asarray_from_seq_single_device(
                obj,
                seq_shape,
                seq_dt,
                seq_dev,
                dtype=dtype,
                usm_type=usm_type,
                sycl_queue=sycl_queue,
                order=order,
            )
        elif len(devs) > 1:
            devs = [dev for dev in devs if dev is not None]
            if sycl_queue is None:
                if len(devs) == 1:
                    alloc_q = devs[0]
                else:
                    raise dpctl.utils.ExecutionPlacementError(
                        "Please specify `device` or `sycl_queue` keyword "
                        "argument to determine where to allocate the "
                        "resulting array."
                    )
            else:
                alloc_q = sycl_queue
            return _asarray_from_seq(
                obj,
                seq_shape,
                seq_dt,
                alloc_q,
                #  force copying via host
                None,
                dtype=dtype,
                usm_type=usm_type,
                order=order,
            )

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
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """ empty(shape, dtype=None, order="C", device=None, \
            usm_type="device", sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` from uninitialized
    USM allocation.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. The `None` value creates an array of floating
            point data type. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            Created empty array.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    _ensure_native_dtype_device_support(dtype, sycl_queue.sycl_device)
    res = dpt.usm_ndarray(
        shape,
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
    if np.issubdtype(dt, np.floating):
        return tuple(float(v) for v in args), dt
    if np.issubdtype(dt, np.complexfloating):
        return tuple(complex(v) for v in args), dt
    if allow_bool and dt.char == "?":
        return tuple(bool(v) for v in args), dt
    raise ValueError(f"Data type {dt} is not supported")


def _round_for_arange(tmp):
    k = int(tmp)
    if k >= 0 and float(k) < tmp:
        tmp = tmp + 1
    return tmp


def _get_arange_length(start, stop, step):
    "Compute length of arange sequence"
    span = stop - start
    if hasattr(step, "__float__") and hasattr(span, "__float__"):
        return _round_for_arange(span / step)
    tmp = span / step
    if hasattr(tmp, "__complex__"):
        tmp = complex(tmp)
        tmp = tmp.real
    else:
        tmp = float(tmp)
    return _round_for_arange(tmp)


def _to_scalar(obj, sc_ty):
    "A way to convert object to NumPy scalar type"
    zd_arr = np.asarray(obj).astype(sc_ty, casting="unsafe")
    return zd_arr[tuple()]


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
               device=None, usm_type="device", sycl_queue=None)

    Returns evenly spaced values within the half-open interval [start, stop)
    as a one-dimensional array.

    Args:
        start: Starting point of the interval
        stop: Ending point of the interval. Default: `None`
        step: Increment of the returned sequence. Default: `1`
        dtype: Output array data type. Default: `None`
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `'device'`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            Array populated with evenly spaced values.
    """
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    is_bool = False
    if dtype:
        is_bool = (dtype is bool) or (dpt.dtype(dtype) == dpt.bool)
    _, dt = _coerce_and_infer_dt(
        start,
        stop,
        step,
        dt=dpt.int8 if is_bool else dtype,
        sycl_queue=sycl_queue,
        err_msg="start, stop, and step must be Python scalars",
        allow_bool=False,
    )
    try:
        tmp = _get_arange_length(start, stop, step)
        sh = max(int(tmp), 0)
    except TypeError:
        sh = 0
    if is_bool and sh > 2:
        raise ValueError("no fill-function for boolean data type")
    res = dpt.usm_ndarray(
        (sh,),
        dtype=dt,
        buffer=usm_type,
        order="C",
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    sc_ty = dt.type
    _first = _to_scalar(start, sc_ty)
    if sh > 1:
        _second = _to_scalar(start + step, sc_ty)
        if dt in [dpt.uint8, dpt.uint16, dpt.uint32, dpt.uint64]:
            int64_ty = dpt.int64.type
            _step = int64_ty(_second) - int64_ty(_first)
        else:
            _step = _second - _first
        _step = sc_ty(_step)
    else:
        _step = sc_ty(1)
    _start = _first
    hev, _ = ti._linspace_step(_start, _step, res, sycl_queue)
    hev.wait()
    if is_bool:
        res_out = dpt.usm_ndarray(
            (sh,),
            dtype=dpt.bool,
            buffer=usm_type,
            order="C",
            buffer_ctor_kwargs={"queue": sycl_queue},
        )
        res_out[:] = res
        res = res_out
    return res


def zeros(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """ zeros(shape, dtype=None, order="C", device=None, \
              usm_type="device", sycl_queue=None)

    Returns a new :class:`dpctl.tensor.usm_ndarray` having a specified
    shape and filled with zeros.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: `"C"`
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            Constructed array initialized with zeros.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    _ensure_native_dtype_device_support(dtype, sycl_queue.sycl_device)
    res = dpt.usm_ndarray(
        shape,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    res.usm_data.memset()
    return res


def ones(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """ ones(shape, dtype=None, order="C", \
             device=None, usm_type="device", sycl_queue=None)

    Returns a new :class:`dpctl.tensor.usm_ndarray` having a specified
    shape and filled with ones.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: `"C"`
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            Created array initialized with ones.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    res = dpt.usm_ndarray(
        shape,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    hev, _ = ti._full_usm_ndarray(1, res, sycl_queue)
    hev.wait()
    return res


def full(
    shape,
    fill_value,
    dtype=None,
    order="C",
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """ full(shape, fill_value, dtype=None, order="C", \
             device=None, usm_type=None, sycl_queue=None)

    Returns a new :class:`dpctl.tensor.usm_ndarray` having a specified
    shape and filled with `fill_value`.

    Args:
        shape (tuple): Dimensions of the array to be created.
        fill_value (int,float,complex): fill value
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host"|None, optional): The type of SYCL
            USM allocation for the output array. If `usm_type` is `None`, it is
            inferred from `fill_value` input if it is an instance of
            `ums_ndarray`, or interpreted as `"device"` otherwise.
            Default: `None`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            New array initialized with given value.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=True)

    if isinstance(fill_value, (dpt.usm_ndarray, np.ndarray, tuple, list)):
        if (
            isinstance(fill_value, dpt.usm_ndarray)
            and sycl_queue is None
            and device is None
        ):
            sycl_queue = fill_value.sycl_queue
        else:
            sycl_queue = normalize_queue_device(
                sycl_queue=sycl_queue, device=device
            )
        X = dpt.asarray(
            fill_value,
            dtype=dtype,
            order=order,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
        return dpt.copy(dpt.broadcast_to(X, shape), order=order)

    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    usm_type = usm_type if usm_type is not None else "device"
    fill_value_type = type(fill_value)
    dtype = _get_dtype(dtype, sycl_queue, ref_type=fill_value_type)
    res = dpt.usm_ndarray(
        shape,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    if fill_value_type in [float, complex] and np.issubdtype(dtype, np.integer):
        fill_value = int(fill_value.real)
    elif fill_value_type is complex and np.issubdtype(dtype, np.floating):
        fill_value = fill_value.real
    elif fill_value_type is int and np.issubdtype(dtype, np.integer):
        fill_value = _to_scalar(fill_value, dtype)

    hev, _ = ti._full_usm_ndarray(fill_value, res, sycl_queue)
    hev.wait()
    return res


def empty_like(
    x, dtype=None, order="C", device=None, usm_type=None, sycl_queue=None
):
    """ empty_like(x, dtype=None, order="C", \
                   device=None, usm_type=None, sycl_queue=None)

    Returns an uninitialized :class:`dpctl.tensor.usm_ndarray` with the
    same `shape` as the input array `x`.

    Args:
        x (usm_ndarray): Input array from which to derive the output array
            shape.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: `"C"`
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host"|None, optional): The type of SYCL
            USM allocation for the output array. If `usm_type` is `None`, the
            the `usm_type` is inferred from the input array. Default: `None`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            Created empty array with uninitialized memory.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    if dtype is None:
        dtype = x.dtype
    if usm_type is None:
        usm_type = x.usm_type
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if device is None and sycl_queue is None:
        device = x.device
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    shape = x.shape
    dtype = dpt.dtype(dtype)
    _ensure_native_dtype_device_support(dtype, sycl_queue.sycl_device)
    res = dpt.usm_ndarray(
        shape,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    return res


def zeros_like(
    x, dtype=None, order="C", device=None, usm_type=None, sycl_queue=None
):
    """ zeros_like(x, dtype=None, order="C", \
                   device=None, usm_type=None, sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` from USM allocation
    initialized with zeros.

    Args:
        x (usm_ndarray): Input array from which to derive the shape of the
            output array.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. If `None`, output array has the same data type as
            the input array. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: `"C"`.
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding
            to a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host"|None, optional): The type of SYCL
            USM allocation for the output array. If `None`, output array has
            the same USM allocation type as the input array. Default: `None`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            New array initialized with zeros.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
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
    """ ones_like(x, dtype=None, order="C", \
                  device=None, usm_type=None, sycl_queue=None)

    Returns a new :class:`dpctl.tensor.usm_ndarray` filled with ones and
    having the same `shape` as the input array `x`.

    Args:
        x (usm_ndarray): Input array from which to derive the output array
            shape.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: `"C"`.
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding
            to a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host"|None, optional): The type of SYCL
            USM allocation for the output array. If `None`, output array has
            the same USM allocation type as the input array. Default: `None`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            New array initialized with ones.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
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
    """ full_like(x, fill_value, dtype=None, order="C", \
                  device=None, usm_type=None, sycl_queue=None)

    Returns a new :class:`dpctl.tensor.usm_ndarray` filled with `fill_value`
    and having the same `shape` as the input array `x`.

    Args:
        x (usm_ndarray): Input array from which to derive the output array
            shape.
        fill_value: the value to fill output array with
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. If `dtype` is `None`, the output array data type is
            inferred from `x`. Default: `None`.
        order ("C", or F"): memory layout for the array. Default: `"C"`.
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host"|None, optional): The type of SYCL
            USM allocation for the output array. If `None`, output array has
            the same USM allocation type as the input array `x`.
            Default: `None`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            New array initialized with given value.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected instance of dpt.usm_ndarray, got {type(x)}.")
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
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
    linspace(start, stop, num, dtype=None, device=None, endpoint=True, \
        sycl_queue=None, usm_type="device")

    Returns :class:`dpctl.tensor.usm_ndarray` array populated with
    evenly spaced numbers of specified interval.

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
            the function raises `ValueError` exception.
        dtype: output array data type. Should be a floating data type.
            If `dtype` is `None`, the output array must be the default
            floating point data type. Default: `None`.
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.
        endpoint: boolean indicating whether to include `stop` in the
            interval. Default: `True`.

    Returns:
        usm_ndarray:
            Array populated with evenly spaced numbers in the requested
            interval.
    """
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    if endpoint not in [True, False]:
        raise TypeError("endpoint keyword argument must be of boolean type")
    num = operator.index(num)
    if num < 0:
        raise ValueError("Number of points must be non-negative")
    _, dt = _coerce_and_infer_dt(
        start,
        stop,
        dt=dtype,
        sycl_queue=sycl_queue,
        err_msg="start and stop must be Python scalars.",
        allow_bool=True,
    )
    int_dt = None
    if np.issubdtype(dt, np.integer):
        if dtype is not None:
            int_dt = dt
        dt = ti.default_device_fp_type(sycl_queue)
        dt = dpt.dtype(dt)
        start = float(start)
        stop = float(stop)
    res = dpt.empty(num, dtype=dt, usm_type=usm_type, sycl_queue=sycl_queue)
    hev, _ = ti._linspace_affine(
        start, stop, dst=res, include_endpoint=endpoint, sycl_queue=sycl_queue
    )
    hev.wait()
    return res if int_dt is None else dpt.astype(res, int_dt)


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
    eye(n_rows, n_cols=None, /, *, k=0, dtype=None, \
        device=None, usm_type="device", sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` with ones on the `k`-th
    diagonal.

    Args:
        n_rows: number of rows in the output array.
        n_cols (optional): number of columns in the output array. If None,
            n_cols = n_rows. Default: `None`.
        k: index of the diagonal, with 0 as the main diagonal.
            A positive value of k is a superdiagonal, a negative value
            is a subdiagonal.
            Raises `TypeError` if `k` is not an integer.
            Default: `0`.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: `None`.
        order ("C" or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returned by
            :attr:`dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both are `None`, a cached
            queue targeting default-selected device is used for allocation
            and copying. Default: `None`.

    Returns:
        usm_ndarray:
            A diagonal matrix.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
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
    _ensure_native_dtype_device_support(dtype, sycl_queue.sycl_device)
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


def tril(x, k=0):
    """
    tril(x, k=0)

    Returns the lower triangular part of a matrix (or a stack of matrices) `x`.

    The lower triangular part of the matrix is defined as the elements on and
    below the specified diagonal `k`.

    Args:
        x (usm_ndarray): Input array.
        k (int, optional): Specifies the diagonal above which to set
            elements to zero. If `k = 0`, the diagonal is the main diagonal.
            If `k < 0`, the diagonal is below the main diagonal.
            If `k > 0`, the diagonal is above the main diagonal. Default: `0`.

    Returns:
        usm_ndarray:
            A lower-triangular array or a stack of lower-triangular arrays.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected argument of type dpctl.tensor.usm_ndarray, "
            f"got {type(x)}."
        )

    k = operator.index(k)

    order = "F" if (x.flags.f_contiguous) else "C"

    shape = x.shape
    nd = x.ndim
    if nd < 2:
        raise ValueError("Array dimensions less than 2.")

    q = x.sycl_queue
    if k >= shape[nd - 1] - 1:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=res, sycl_queue=q
        )
        hev.wait()
    elif k < -shape[nd - 2]:
        res = dpt.zeros(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
    else:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        hev, _ = ti._tril(src=x, dst=res, k=k, sycl_queue=q)
        hev.wait()

    return res


def triu(x, k=0):
    """
    triu(x, k=0)

    Returns the upper triangular part of a matrix (or a stack of matrices) `x`.

    The upper triangular part of the matrix is defined as the elements on and
    above the specified diagonal `k`.

    Args:
        x (usm_ndarray): Input array.
        k (int, optional): Specifies the diagonal below which to set
            elements to zero. If `k = 0`, the diagonal is the main diagonal.
            If `k < 0`, the diagonal is below the main diagonal.
            If `k > 0`, the diagonal is above the main diagonal. Default: `0`.

    Returns:
        usm_ndarray:
            An upper-triangular array or a stack of upper-triangular arrays.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected argument of type dpctl.tensor.usm_ndarray, "
            f"got {type(x)}."
        )

    k = operator.index(k)

    order = "F" if (x.flags.f_contiguous) else "C"

    shape = x.shape
    nd = x.ndim
    if nd < 2:
        raise ValueError("Array dimensions less than 2.")

    q = x.sycl_queue
    if k > shape[nd - 1]:
        res = dpt.zeros(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
    elif k <= -shape[nd - 2] + 1:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=x, dst=res, sycl_queue=q
        )
        hev.wait()
    else:
        res = dpt.empty(
            x.shape,
            dtype=x.dtype,
            order=order,
            usm_type=x.usm_type,
            sycl_queue=q,
        )
        hev, _ = ti._triu(src=x, dst=res, k=k, sycl_queue=q)
        hev.wait()

    return res


def meshgrid(*arrays, indexing="xy"):
    """meshgrid(*arrays, indexing="xy")

    Creates list of :class:`dpctl.tensor.usm_ndarray` coordinate matrices
    from vectors.

    Args:
        arrays (usm_ndarray): an arbitrary number of one-dimensional arrays
            representing grid coordinates. Each array should have the same
            numeric data type.
        indexing: Cartesian (`xy`) or matrix (`ij`) indexing of output.
            If provided zero or one one-dimensional vector(s) (i.e., the
            zero- and one-dimensional cases, respectively), the `indexing`
            keyword has no effect and should be ignored. Default: `xy`.

    Returns:
        List[array]: list of `N` arrays, where `N` is the number of
            provided one-dimensional input arrays. Each returned array must
            have rank `N`.
            For a set of `n` vectors with lengths `N0`, `N1`, `N2`, ...
            The cartesian indexing results in arrays of shape
            `(N1, N0, N2, ...)`, while the
            matrix indexing results in arrays of shape
            `(N0, N1, N2, ...)`.
            Default: `xy`.

    Raises:
        ValueError: If vectors are not of the same data type, or are not
            one-dimensional.

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
    if n == 0:
        return []

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
