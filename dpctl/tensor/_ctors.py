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

import numpy as np

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
import dpctl.utils

_empty_tuple = tuple()
_host_set = frozenset([None])


def _array_info_dispatch(obj):
    if isinstance(obj, dpt.usm_ndarray):
        return obj.shape, obj.dtype, frozenset([obj.sycl_queue])
    elif isinstance(obj, np.ndarray):
        return obj.shape, obj.dtype, _host_set
    elif isinstance(obj, range):
        return (len(obj),), int, _host_set
    elif isinstance(obj, float):
        return _empty_tuple, float, _host_set
    elif isinstance(obj, int):
        return _empty_tuple, int, _host_set
    elif isinstance(obj, complex):
        return _empty_tuple, complex, _host_set
    elif isinstance(obj, (list, tuple, range)):
        return _array_info_sequence(obj)
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
        device = _host_set
    return (n,) + dim, dt, device


def _normalize_queue_device(q=None, d=None):
    if q is None:
        d = dpt._device.Device.create_device(d)
        return d.sycl_queue
    else:
        if not isinstance(q, dpctl.SyclQueue):
            raise TypeError(f"Expected dpctl.SyclQueue, got {type(q)}")
        if d is None:
            return q
        d = dpt._device.Device.create_device(d)
        qq = dpctl.utils.get_execution_queue(
            (
                q,
                d.sycl_queue,
            )
        )
        if qq is None:
            raise TypeError(
                "sycl_queue and device keywords can not be both specified"
            )
        return qq


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
        copy_q = _normalize_queue_device(q=sycl_queue, d=exec_q)
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
    c_contig = usm_ndary.flags & 1
    f_contig = usm_ndary.flags & 2
    fc_contig = usm_ndary.flags & 3
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
    if dtype is None:
        dtype = ary.dtype
    copy_q = _normalize_queue_device(q=None, d=sycl_queue)
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
    """asarray(obj, dtype=None, copy=None, order="K",
               device=None, usm_type=None, sycl_queue=None)

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
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. For `usm_type=None` the allocation
            type is inferred from the input if `obj` has USM allocation, or
            `"device"` is used instead. Default: `None`.
        sycl_queue: (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both a `None`, the
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
        dtype = np.dtype(dtype)
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
    if usm_type is not None:
        if isinstance(usm_type, str):
            if usm_type not in ["device", "shared", "host"]:
                raise ValueError(
                    f"Unrecognized value of usm_type={usm_type}, "
                    "expected 'device', 'shared', 'host', or None."
                )
        else:
            raise TypeError(
                f"Expected usm_type to be a str or None, got {type(usm_type)}"
            )
    # 5. Normalize device/sycl_queue [keep it None if was None]
    if device is not None or sycl_queue is not None:
        sycl_queue = _normalize_queue_device(q=sycl_queue, d=device)

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
                np.asarray(obj, dt, order=order),
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
    sh, dtype="f8", order="C", device=None, usm_type="device", sycl_queue=None
):
    """dpctl.tensor.empty(shape, dtype="f8", order="C", device=None,
                          usm_type="device", sycl_queue=None)

    Creates `usm_ndarray` from uninitialized USM allocation.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (optional): data type of the array. Can be typestring,
            a `numpy.dtype` object, `numpy` char string, or a numpy
            scalar type. Default: "f8"
        order ("C", or F"): memory layout for the array. Default: "C"
        device (optional): array API concept of device where the output array
            is created. `device` can be `None`, a oneAPI filter selector string,
            an instance of :class:`dpctl.SyclDevice` corresponding to a
            non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `None`.
        usm_type ("device"|"shared"|"host", optional): The type of SYCL USM
            allocation for the output array. Default: `"device"`.
        sycl_queue: (:class:`dpctl.SyclQueue`, optional): The SYCL queue to use
            for output array allocation and copying. `sycl_queue` and `device`
            are exclusive keywords, i.e. use one or another. If both are
            specified, a `TypeError` is raised unless both imply the same
            underlying SYCL queue to be used. If both a `None`, the
            `dpctl.SyclQueue()` is used for allocation and copying.
            Default: `None`.
    """
    dtype = np.dtype(dtype)
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    else:
        order = order[0].upper()
    if isinstance(usm_type, str):
        if usm_type not in ["device", "shared", "host"]:
            raise ValueError(
                f"Unrecognized value of usm_type={usm_type}, "
                "expected 'device', 'shared', or 'host'."
            )
    else:
        raise TypeError(
            f"Expected usm_type to be of type str, got {type(usm_type)}"
        )
    sycl_queue = _normalize_queue_device(q=sycl_queue, d=device)
    res = dpt.usm_ndarray(
        sh,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    return res
