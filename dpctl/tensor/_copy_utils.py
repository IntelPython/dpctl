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
from numpy.core.numeric import normalize_axis_index

import dpctl
import dpctl.memory as dpm
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import dpctl.utils
from dpctl.tensor._device import normalize_queue_device

__doc__ = (
    "Implementation module for copy- and cast- operations on "
    ":class:`dpctl.tensor.usm_ndarray`."
)


def _has_memory_overlap(x1, x2):
    if x1.size and x2.size:
        m1 = dpm.as_usm_memory(x1)
        m2 = dpm.as_usm_memory(x2)
        # can only overlap if bound to the same context
        if m1.sycl_context == m2.sycl_context:
            p1_beg = m1._pointer
            p1_end = p1_beg + m1.nbytes
            p2_beg = m2._pointer
            p2_end = p2_beg + m2.nbytes
            # may intersect if not ((p1_beg >= p2_end) or (p2_beg >= p2_end))
            return (p1_beg < p2_end) and (p2_beg < p1_end)
        return False
    # zero element array do not overlap anything
    return False


def _copy_to_numpy(ary):
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError
    nb = ary.usm_data.nbytes
    hh = dpm.MemoryUSMHost(nb, queue=ary.sycl_queue)
    hh.copy_from_device(ary.usm_data)
    h = np.ndarray(nb, dtype="u1", buffer=hh).view(ary.dtype)
    itsz = ary.itemsize
    strides_bytes = tuple(si * itsz for si in ary.strides)
    offset = ary.__sycl_usm_array_interface__.get("offset", 0) * itsz
    return np.ndarray(
        ary.shape,
        dtype=ary.dtype,
        buffer=h,
        strides=strides_bytes,
        offset=offset,
    )


def _copy_from_numpy(np_ary, usm_type="device", sycl_queue=None):
    "Copies numpy array `np_ary` into a new usm_ndarray"
    # This may peform a copy to meet stated requirements
    Xnp = np.require(np_ary, requirements=["A", "E"])
    alloc_q = normalize_queue_device(sycl_queue=sycl_queue, device=None)
    dt = Xnp.dtype
    if dt.char in "dD" and alloc_q.sycl_device.has_aspect_fp64 is False:
        Xusm_dtype = (
            dpt.dtype("float32") if dt.char == "d" else dpt.dtype("complex64")
        )
    else:
        Xusm_dtype = dt
    Xusm = dpt.empty(
        Xnp.shape, dtype=Xusm_dtype, usm_type=usm_type, sycl_queue=sycl_queue
    )
    _copy_from_numpy_into(Xusm, Xnp)
    return Xusm


def _copy_from_numpy_into(dst, np_ary):
    "Copies `np_ary` into `dst` of type :class:`dpctl.tensor.usm_ndarray"
    if not isinstance(np_ary, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(np_ary)}")
    if not isinstance(dst, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(dst)}")
    if np_ary.flags["OWNDATA"]:
        Xnp = np_ary
    else:
        # Determine base of input array
        base = np_ary.base
        while isinstance(base, np.ndarray):
            base = base.base
        if isinstance(base, dpm._memory._Memory):
            # we must perform a copy, since subsequent
            # _copy_numpy_ndarray_into_usm_ndarray is implemented using
            # sycl::buffer, and using USM-pointers with sycl::buffer
            # results is undefined behavior
            Xnp = np_ary.copy()
        else:
            Xnp = np_ary
    src_ary = np.broadcast_to(Xnp, dst.shape)
    copy_q = dst.sycl_queue
    if copy_q.sycl_device.has_aspect_fp64 is False:
        src_ary_dt_c = src_ary.dtype.char
        if src_ary_dt_c == "d":
            src_ary = src_ary.astype(np.float32)
        elif src_ary_dt_c == "D":
            src_ary = src_ary.astype(np.complex64)
    ti._copy_numpy_ndarray_into_usm_ndarray(
        src=src_ary, dst=dst, sycl_queue=copy_q
    )


def from_numpy(np_ary, device=None, usm_type="device", sycl_queue=None):
    """
    from_numpy(arg, device=None, usm_type="device", sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` from instance of
    `numpy.ndarray`.

    Args:
        arg: An instance of input `numpy.ndarray`
        device: array API specification of device where the output array
            is created
        usm_type: The requested USM allocation type for the output array
        sycl_queue: a :class:`dpctl.SyclQueue` instance that determines
            output array allocation device as well as placement of data
            movement operation. The `device` and `sycl_queue` arguments
            are equivalent. Only one of them should be specified. If both
            are provided, they must be consistent and result in using the
            same execution queue.

    The returned array has the same shape, and the same data type kind.
    If the device does not support the data type of input array, a
    closest support data type of the same kind may be returned, e.g.
    input array of type `float16` may be upcast to `float32` if the
    target device does not support 16-bit floating point type.
    """
    q = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    return _copy_from_numpy(np_ary, usm_type=usm_type, sycl_queue=q)


def to_numpy(usm_ary):
    """
    to_numpy(usm_ary)

    Copies content of :class:`dpctl.tensor.usm_ndarray` instance `usm_ary`
    into `numpy.ndarray` instance of the same shape and same data type.

    Args:
        usm_ary: An instance of :class:`dpctl.tensor.usm_ndarray`
    Returns:
        An instance of `numpy.ndarray` populated with content of `usm_ary`.
    """
    return _copy_to_numpy(usm_ary)


def asnumpy(usm_ary):
    """
    asnumpy(usm_ary)

    Copies content of :class:`dpctl.tensor.usm_ndarray` instance `usm_ary`
    into `numpy.ndarray` instance of the same shape and same data type.

    Args:
        usm_ary: An instance of :class:`dpctl.tensor.usm_ndarray`
    Returns:
        An instance of `numpy.ndarray` populated with content of `usm_ary`.
    """
    return _copy_to_numpy(usm_ary)


class Dummy:
    "Helper class with specified __sycl_usm_array_interface__ attribute"

    def __init__(self, iface):
        self.__sycl_usm_array_interface__ = iface


def _copy_overlapping(dst, src):
    """Assumes src and dst have the same shape."""
    q = normalize_queue_device(sycl_queue=dst.sycl_queue)
    tmp = dpt.usm_ndarray(
        src.shape,
        dtype=src.dtype,
        buffer="device",
        order="C",
        buffer_ctor_kwargs={"queue": q},
    )
    hcp1, cp1 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=src, dst=tmp, sycl_queue=q
    )
    hcp2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
        src=tmp, dst=dst, sycl_queue=q, depends=[cp1]
    )
    hcp2.wait()
    hcp1.wait()


def _copy_same_shape(dst, src):
    """Assumes src and dst have the same shape."""
    # check that memory regions do not overlap
    if _has_memory_overlap(dst, src):
        _copy_overlapping(src=src, dst=dst)
        return

    hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
        src=src, dst=dst, sycl_queue=dst.sycl_queue
    )
    hev.wait()


if hasattr(np, "broadcast_shapes"):

    def _broadcast_shapes(sh1, sh2):
        return np.broadcast_shapes(sh1, sh2)

else:

    def _broadcast_shapes(sh1, sh2):
        # use arrays with zero strides, whose memory footprint
        # is independent of the number of array elements
        return np.broadcast(
            np.empty(sh1, dtype=[]),
            np.empty(sh2, dtype=[]),
        ).shape


def _copy_from_usm_ndarray_to_usm_ndarray(dst, src):
    if any(
        not isinstance(arg, dpt.usm_ndarray)
        for arg in (
            dst,
            src,
        )
    ):
        raise TypeError(
            "Both types are expected to be dpctl.tensor.usm_ndarray, "
            f"got {type(dst)} and {type(src)}."
        )

    if dst.ndim == src.ndim and dst.shape == src.shape:
        _copy_same_shape(dst, src)
        return

    try:
        common_shape = _broadcast_shapes(dst.shape, src.shape)
    except ValueError as exc:
        raise ValueError("Shapes of two arrays are not compatible") from exc

    if dst.size < src.size:
        raise ValueError("Destination is smaller ")

    if len(common_shape) > dst.ndim:
        ones_count = len(common_shape) - dst.ndim
        for k in range(ones_count):
            if common_shape[k] != 1:
                raise ValueError
        common_shape = common_shape[ones_count:]

    if src.ndim < len(common_shape):
        new_src_strides = (0,) * (len(common_shape) - src.ndim) + src.strides
        src_same_shape = dpt.usm_ndarray(
            common_shape, dtype=src.dtype, buffer=src, strides=new_src_strides
        )
    else:
        src_same_shape = src
        src_same_shape.shape = common_shape

    _copy_same_shape(dst, src_same_shape)


def copy(usm_ary, order="K"):
    """
    Creates a copy of given instance of `usm_ndarray`.

    Memory layour of the copy is controlled by `order` keyword,
    following NumPy's conventions. The `order` keywords can be
    one of the following:

       - "C": C-contiguous memory layout
       - "F": Fortran-contiguous memory layout
       - "A": Fortran-contiguous if the input array is also Fortran-contiguous,
         otherwise C-contiguous
       - "K": match the layout of `usm_ary` as closely as possible.

    """
    if not isinstance(usm_ary, dpt.usm_ndarray):
        return TypeError(
            f"Expected object of type dpt.usm_ndarray, got {type(usm_ary)}"
        )
    copy_order = "C"
    if order == "C":
        pass
    elif order == "F":
        copy_order = order
    elif order == "A":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    elif order == "K":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    else:
        raise ValueError(
            "Unrecognized value of the order keyword. "
            "Recognized values are 'A', 'C', 'F', or 'K'"
        )
    c_contig = usm_ary.flags.c_contiguous
    f_contig = usm_ary.flags.f_contiguous
    R = dpt.usm_ndarray(
        usm_ary.shape,
        dtype=usm_ary.dtype,
        buffer=usm_ary.usm_type,
        order=copy_order,
        buffer_ctor_kwargs={"queue": usm_ary.sycl_queue},
    )
    if order == "K" and (not c_contig and not f_contig):
        original_strides = usm_ary.strides
        ind = sorted(
            range(usm_ary.ndim),
            key=lambda i: abs(original_strides[i]),
            reverse=True,
        )
        new_strides = tuple(R.strides[ind[i]] for i in ind)
        R = dpt.usm_ndarray(
            usm_ary.shape,
            dtype=usm_ary.dtype,
            buffer=R.usm_data,
            strides=new_strides,
        )
    _copy_same_shape(R, usm_ary)
    return R


def astype(usm_ary, newdtype, order="K", casting="unsafe", copy=True):
    """
    astype(usm_array, new_dtype, order="K", casting="unsafe", copy=True)

    Returns a copy of the array, cast to a specified type.

    A view can be returned, if possible, when `copy=False` is used.
    """
    if not isinstance(usm_ary, dpt.usm_ndarray):
        return TypeError(
            f"Expected object of type dpt.usm_ndarray, got {type(usm_ary)}"
        )
    if not isinstance(order, str) or order not in ["A", "C", "F", "K"]:
        raise ValueError(
            "Unrecognized value of the order keyword. "
            "Recognized values are 'A', 'C', 'F', or 'K'"
        )
    ary_dtype = usm_ary.dtype
    target_dtype = dpt.dtype(newdtype)
    if not dpt.can_cast(ary_dtype, target_dtype, casting=casting):
        raise TypeError(
            f"Can not cast from {ary_dtype} to {newdtype} "
            f"according to rule {casting}."
        )
    c_contig = usm_ary.flags.c_contiguous
    f_contig = usm_ary.flags.f_contiguous
    needs_copy = copy or not ary_dtype == target_dtype
    if not needs_copy and (order != "K"):
        needs_copy = (c_contig and order not in ["A", "C"]) or (
            f_contig and order not in ["A", "F"]
        )
    if not needs_copy:
        return usm_ary
    copy_order = "C"
    if order == "C":
        pass
    elif order == "F":
        copy_order = order
    elif order == "A":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    elif order == "K":
        if usm_ary.flags.f_contiguous:
            copy_order = "F"
    else:
        raise ValueError(
            "Unrecognized value of the order keyword. "
            "Recognized values are 'A', 'C', 'F', or 'K'"
        )
    R = dpt.usm_ndarray(
        usm_ary.shape,
        dtype=target_dtype,
        buffer=usm_ary.usm_type,
        order=copy_order,
        buffer_ctor_kwargs={"queue": usm_ary.sycl_queue},
    )
    if order == "K" and (not c_contig and not f_contig):
        original_strides = usm_ary.strides
        ind = sorted(
            range(usm_ary.ndim),
            key=lambda i: abs(original_strides[i]),
            reverse=True,
        )
        new_strides = tuple(R.strides[ind[i]] for i in ind)
        R = dpt.usm_ndarray(
            usm_ary.shape,
            dtype=target_dtype,
            buffer=R.usm_data,
            strides=new_strides,
        )
    _copy_from_usm_ndarray_to_usm_ndarray(R, usm_ary)
    return R


def _extract_impl(ary, ary_mask, axis=0):
    """Extract elements of ary by applying mask starting from slot
    dimension axis"""
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary)}"
        )
    if not isinstance(ary_mask, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary_mask)}"
        )
    exec_q = dpctl.utils.get_execution_queue(
        (ary.sycl_queue, ary_mask.sycl_queue)
    )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "arrays have different associated queues. "
            "Use `Y.to_device(X.device)` to migrate."
        )
    ary_nd = ary.ndim
    pp = normalize_axis_index(operator.index(axis), ary_nd)
    mask_nd = ary_mask.ndim
    if pp < 0 or pp + mask_nd > ary_nd:
        raise ValueError(
            "Parameter p is inconsistent with input array dimensions"
        )
    mask_nelems = ary_mask.size
    cumsum = dpt.empty(mask_nelems, dtype=dpt.int64, device=ary_mask.device)
    exec_q = cumsum.sycl_queue
    mask_count = ti.mask_positions(ary_mask, cumsum, sycl_queue=exec_q)
    dst_shape = ary.shape[:pp] + (mask_count,) + ary.shape[pp + mask_nd :]
    dst = dpt.empty(
        dst_shape, dtype=ary.dtype, usm_type=ary.usm_type, device=ary.device
    )
    hev, _ = ti._extract(
        src=ary,
        cumsum=cumsum,
        axis_start=pp,
        axis_end=pp + mask_nd,
        dst=dst,
        sycl_queue=exec_q,
    )
    hev.wait()
    return dst


def _nonzero_impl(ary):
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary)}"
        )
    exec_q = ary.sycl_queue
    usm_type = ary.usm_type
    mask_nelems = ary.size
    cumsum = dpt.empty(
        mask_nelems, dtype=dpt.int64, sycl_queue=exec_q, order="C"
    )
    mask_count = ti.mask_positions(ary, cumsum, sycl_queue=exec_q)
    indexes = dpt.empty(
        (ary.ndim, mask_count),
        dtype=cumsum.dtype,
        usm_type=usm_type,
        sycl_queue=exec_q,
        order="C",
    )
    hev, _ = ti._nonzero(cumsum, indexes, ary.shape, exec_q)
    res = tuple(indexes[i, :] for i in range(ary.ndim))
    hev.wait()
    return res


def _take_multi_index(ary, inds, p):
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError
    queues_ = [
        ary.sycl_queue,
    ]
    usm_types_ = [
        ary.usm_type,
    ]
    if not isinstance(inds, list) and not isinstance(inds, tuple):
        inds = (inds,)
    all_integers = True
    for ind in inds:
        queues_.append(ind.sycl_queue)
        usm_types_.append(ind.usm_type)
        if all_integers:
            all_integers = ind.dtype.kind in "ui"
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError("")
    if not all_integers:
        raise IndexError(
            "arrays used as indices must be of integer (or boolean) type"
        )
    if len(inds) > 1:
        inds = dpt.broadcast_arrays(*inds)
    ary_ndim = ary.ndim
    p = normalize_axis_index(operator.index(p), ary_ndim)

    res_shape = ary.shape[:p] + inds[0].shape + ary.shape[p + len(inds) :]
    res_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)
    res = dpt.empty(
        res_shape, dtype=ary.dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    hev, _ = ti._take(
        src=ary, ind=inds, dst=res, axis_start=p, mode=0, sycl_queue=exec_q
    )
    hev.wait()

    return res


def _place_impl(ary, ary_mask, vals, axis=0):
    """Extract elements of ary by applying mask starting from slot
    dimension axis"""
    if not isinstance(ary, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary)}"
        )
    if not isinstance(ary_mask, dpt.usm_ndarray):
        raise TypeError(
            f"Expecting type dpctl.tensor.usm_ndarray, got {type(ary_mask)}"
        )
    exec_q = dpctl.utils.get_execution_queue(
        (
            ary.sycl_queue,
            ary_mask.sycl_queue,
        )
    )
    if exec_q is not None:
        if not isinstance(vals, dpt.usm_ndarray):
            vals = dpt.asarray(vals, dtype=ary.dtype, sycl_queue=exec_q)
        else:
            exec_q = dpctl.utils.get_execution_queue((exec_q, vals.sycl_queue))
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "arrays have different associated queues. "
            "Use `Y.to_device(X.device)` to migrate."
        )
    ary_nd = ary.ndim
    pp = normalize_axis_index(operator.index(axis), ary_nd)
    mask_nd = ary_mask.ndim
    if pp < 0 or pp + mask_nd > ary_nd:
        raise ValueError(
            "Parameter p is inconsistent with input array dimensions"
        )
    mask_nelems = ary_mask.size
    cumsum = dpt.empty(mask_nelems, dtype=dpt.int64, device=ary_mask.device)
    exec_q = cumsum.sycl_queue
    mask_count = ti.mask_positions(ary_mask, cumsum, sycl_queue=exec_q)
    expected_vals_shape = (
        ary.shape[:pp] + (mask_count,) + ary.shape[pp + mask_nd :]
    )
    if vals.dtype == ary.dtype:
        rhs = vals
    else:
        rhs = dpt.astype(vals, ary.dtype)
    rhs = dpt.broadcast_to(rhs, expected_vals_shape)
    hev, _ = ti._place(
        dst=ary,
        cumsum=cumsum,
        axis_start=pp,
        axis_end=pp + mask_nd,
        rhs=rhs,
        sycl_queue=exec_q,
    )
    hev.wait()
    return


def _put_multi_index(ary, inds, p, vals):
    if isinstance(vals, dpt.usm_ndarray):
        queues_ = [ary.sycl_queue, vals.sycl_queue]
        usm_types_ = [ary.usm_type, vals.usm_type]
    else:
        queues_ = [
            ary.sycl_queue,
        ]
        usm_types_ = [
            ary.usm_type,
        ]
    if not isinstance(inds, list) and not isinstance(inds, tuple):
        inds = (inds,)
    all_integers = True
    for ind in inds:
        if not isinstance(ind, dpt.usm_ndarray):
            raise TypeError
        queues_.append(ind.sycl_queue)
        usm_types_.append(ind.usm_type)
        if all_integers:
            all_integers = ind.dtype.kind in "ui"
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "Can not automatically determine where to allocate the "
            "result or performance execution. "
            "Use `usm_ndarray.to_device` method to migrate data to "
            "be associated with the same queue."
        )
    if not all_integers:
        raise IndexError(
            "arrays used as indices must be of integer (or boolean) type"
        )
    if len(inds) > 1:
        inds = dpt.broadcast_arrays(*inds)
    ary_ndim = ary.ndim

    p = normalize_axis_index(operator.index(p), ary_ndim)
    vals_shape = ary.shape[:p] + inds[0].shape + ary.shape[p + len(inds) :]

    vals_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)
    if not isinstance(vals, dpt.usm_ndarray):
        vals = dpt.asarray(
            vals, ary.dtype, usm_type=vals_usm_type, sycl_queue=exec_q
        )

    vals = dpt.broadcast_to(vals, vals_shape)

    hev, _ = ti._put(
        dst=ary, ind=inds, val=vals, axis_start=p, mode=0, sycl_queue=exec_q
    )
    hev.wait()

    return
