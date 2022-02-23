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
import operator

import numpy as np

import dpctl.memory as dpm
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
from dpctl.tensor._device import normalize_queue_device


def contract_iter2(shape, strides1, strides2):
    p = np.argsort(np.abs(strides1))[::-1]
    sh = [operator.index(shape[i]) for i in p]
    disp1 = 0
    disp2 = 0
    st1 = []
    st2 = []
    contractable = True
    for i in p:
        this_stride1 = operator.index(strides1[i])
        this_stride2 = operator.index(strides2[i])
        if this_stride1 < 0 and this_stride2 < 0:
            disp1 += this_stride1 * (shape[i] - 1)
            this_stride1 = -this_stride1
            disp2 += this_stride2 * (shape[i] - 1)
            this_stride2 = -this_stride2
        if this_stride1 < 0 or this_stride2 < 0:
            contractable = False
        st1.append(this_stride1)
        st2.append(this_stride2)
    while contractable:
        changed = False
        k = len(sh) - 1
        for i in range(k):
            step1 = st1[i + 1]
            jump1 = st1[i] - (sh[i + 1] - 1) * step1
            step2 = st2[i + 1]
            jump2 = st2[i] - (sh[i + 1] - 1) * step2
            if jump1 == step1 and jump2 == step2:
                changed = True
                st1[i:-1] = st1[i + 1 :]
                st2[i:-1] = st2[i + 1 :]
                sh[i] *= sh[i + 1]
                sh[i + 1 : -1] = sh[i + 2 :]
                sh = sh[:-1]
                st1 = st1[:-1]
                st2 = st2[:-1]
                break
        if not changed:
            break
    return (sh, st1, disp1, st2, disp2)


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
        else:
            return False
    else:
        # zero element array do not overlap anything
        return False


def _copy_to_numpy(ary):
    if type(ary) is not dpt.usm_ndarray:
        raise TypeError
    h = ary.usm_data.copy_to_host().view(ary.dtype)
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
    Xnp = np.require(np_ary, requirements=["A", "O", "C", "E"])
    if sycl_queue:
        ctor_kwargs = {"queue": sycl_queue}
    else:
        ctor_kwargs = dict()
    Xusm = dpt.usm_ndarray(
        Xnp.shape,
        dtype=Xnp.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs=ctor_kwargs,
    )
    Xusm.usm_data.copy_from_host(Xnp.reshape((-1)).view("u1"))
    return Xusm


def _copy_from_numpy_into(dst, np_ary):
    "Copies `np_ary` into `dst` of type :class:`dpctl.tensor.usm_ndarray"
    if not isinstance(np_ary, np.ndarray):
        raise TypeError("Expected numpy.ndarray, got {}".format(type(np_ary)))
    src_ary = np.broadcast_to(np.asarray(np_ary, dtype=dst.dtype), dst.shape)
    if src_ary.size and (dst.flags & 1) and src_ary.flags["C"]:
        dpm.as_usm_memory(dst).copy_from_host(src_ary.reshape((-1,)).view("u1"))
        return
    if src_ary.size and (dst.flags & 2) and src_ary.flags["F"]:
        dpm.as_usm_memory(dst).copy_from_host(src_ary.reshape((-1,)).view("u1"))
        return
    for i in range(dst.size):
        mi = np.unravel_index(i, dst.shape)
        host_buf = np.array(src_ary[mi], ndmin=1).view("u1")
        usm_mem = dpm.as_usm_memory(dst[mi])
        usm_mem.copy_from_host(host_buf)


def from_numpy(np_ary, device=None, usm_type="device", sycl_queue=None):
    """
    from_numpy(arg, device=None, usm_type="device", sycl_queue=None)

    Creates :class:`dpctl.tensor.usm_ndarray` from instance of
    `numpy.ndarray`.

    Args:
        arg: An instance of `numpy.ndarray`
        device: array API specification of device where the output array
            is created.
        sycl_queue: a :class:`dpctl.SyclQueue` used to create the output
            array is created
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
    hcp1, cp1 = ti._copy_usm_ndarray_into_usm_ndarray(src=src, dst=tmp, queue=q)
    hcp2, cp2 = ti._copy_usm_ndarray_into_usm_ndarray(
        src=tmp, dst=dst, queue=q, depends=[cp1]
    )
    hcp2.wait()
    hcp1.wait()


def _copy_same_shape(dst, src):
    """Assumes src and dst have the same shape."""
    # check that memory regions do not overlap
    if _has_memory_overlap(dst, src):
        _copy_overlapping(src=src, dst=dst)
        return

    hev, ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=src, dst=dst, queue=dst.sycl_queue
    )
    hev.wait()


def _copy_from_usm_ndarray_to_usm_ndarray(dst, src):
    if type(dst) is not dpt.usm_ndarray or type(src) is not dpt.usm_ndarray:
        raise TypeError(
            "Both types are expected to be dpctl.tensor.usm_ndarray, "
            f"got {type(dst)} and {type(src)}."
        )

    if dst.ndim == src.ndim and dst.shape == src.shape:
        _copy_same_shape(dst, src)

    try:
        common_shape = np.broadcast_shapes(dst.shape, src.shape)
    except ValueError:
        raise ValueError("Shapes of two arrays are not compatible")

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
            "Expected object of type dpt.usm_ndarray, got {}".format(
                type(usm_ary)
            )
        )
    copy_order = "C"
    if order == "C":
        pass
    elif order == "F":
        copy_order = order
    elif order == "A":
        if usm_ary.flags & 2:
            copy_order = "F"
    elif order == "K":
        if usm_ary.flags & 2:
            copy_order = "F"
    else:
        raise ValueError(
            "Unrecognized value of the order keyword. "
            "Recognized values are 'A', 'C', 'F', or 'K'"
        )
    c_contig = usm_ary.flags & 1
    f_contig = usm_ary.flags & 2
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
            "Expected object of type dpt.usm_ndarray, got {}".format(
                type(usm_ary)
            )
        )
    if not isinstance(order, str) or order not in ["A", "C", "F", "K"]:
        raise ValueError(
            "Unrecognized value of the order keyword. "
            "Recognized values are 'A', 'C', 'F', or 'K'"
        )
    ary_dtype = usm_ary.dtype
    target_dtype = np.dtype(newdtype)
    if not np.can_cast(ary_dtype, target_dtype, casting=casting):
        raise TypeError(
            "Can not cast from {} to {} according to rule {}".format(
                ary_dtype, newdtype, casting
            )
        )
    c_contig = usm_ary.flags & 1
    f_contig = usm_ary.flags & 2
    needs_copy = copy or not (ary_dtype == target_dtype)
    if not needs_copy and (order != "K"):
        needs_copy = (c_contig and order not in ["A", "C"]) or (
            f_contig and order not in ["A", "F"]
        )
    if needs_copy:
        copy_order = "C"
        if order == "C":
            pass
        elif order == "F":
            copy_order = order
        elif order == "A":
            if usm_ary.flags & 2:
                copy_order = "F"
        elif order == "K":
            if usm_ary.flags & 2:
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
    else:
        return usm_ary
