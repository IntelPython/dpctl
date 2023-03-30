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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

import sys

import numpy as np

import dpctl
import dpctl.memory as dpmem

from ._data_types import bool as dpt_bool
from ._device import Device
from ._print import usm_ndarray_repr, usm_ndarray_str

from cpython.mem cimport PyMem_Free
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem

cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpmem
cimport dpctl.tensor._dlpack as c_dlpack

import dpctl.tensor._flags as _flags

include "_stride_utils.pxi"
include "_types.pxi"
include "_slicing.pxi"


def _dispatch_unary_elementwise(ary, name):
    try:
        mod = ary.__array_namespace__()
    except AttributeError:
        return NotImplemented
    if mod is None and "dpnp" in sys.modules:
        fn = getattr(sys.modules["dpnp"], name)
        if callable(fn):
            return fn(ary)
    elif hasattr(mod, name):
        fn = getattr(mod, name)
        if callable(fn):
            return fn(ary)

    return NotImplemented


def _dispatch_binary_elementwise(ary, name, other):
    try:
        mod = ary.__array_namespace__()
    except AttributeError:
        return NotImplemented
    if mod is None and "dpnp" in sys.modules:
        fn = getattr(sys.modules["dpnp"], name)
        if callable(fn):
            return fn(ary, other)
    elif hasattr(mod, name):
        fn = getattr(mod, name)
        if callable(fn):
            return fn(ary, other)

    return NotImplemented


def _dispatch_binary_elementwise2(other, name, ary):
    try:
        mod = ary.__array_namespace__()
    except AttributeError:
        return NotImplemented
    mod = ary.__array_namespace__()
    if mod is None and "dpnp" in sys.modules:
        fn = getattr(sys.modules["dpnp"], name)
        if callable(fn):
            return fn(other, ary)
    elif hasattr(mod, name):
        fn = getattr(mod, name)
        if callable(fn):
            return fn(other, ary)

    return NotImplemented


cdef class InternalUSMArrayError(Exception):
    """
    A InternalError exception is raised when internal
    inconsistency has been detected.
    """
    pass


cdef class usm_ndarray:
    """ usm_ndarray(shape, dtype="|f8", strides=None, buffer="device", \
           offset=0, order="C", buffer_ctor_kwargs=dict(), \
           array_namespace=None)

    An array object represents a multidimensional tensor of numeric
    elements stored in a USM allocation on a SYCL device.

    Arg:
        shape (int, tuple):
            Shape of the array to be created.
        dtype (str, dtype):
            Array data type, i.e. the type of array elements.
            The supported types are
               * ``bool``
                     boolean type
               * ``int8``, ``int16``, ``int32``, ``int64``,
                     signed integer types
               * ``uint8``, ``uint16``, ``uint32``, ``uint64``,
                     unsigned integer types
               * ``float16``
                     half-precision floating type,
                     supported if target device's property
                     ``has_aspect_fp16`` is ``True``
               * ``float32``, ``complex64``
                     single-precision real and complex floating
                     types
               * ``float64``, ``complex128``
                     double-precision real and complex floating
                     types, supported if target device's property
                     ``has_aspect_fp64`` is ``True``.
            Default: ``"|f8"``.
        strides (tuple, optional):
            Strides of the array to be created in elements.
            If ``strides`` has the value ``None``, it is determined by the
            ``shape`` of the array and the requested ``order``.
            Default: ``None``.
        buffer (str, object, optional):
            A string corresponding to the type of USM allocation to make,
            or a Python object representing a USM memory allocation, i.e.
            :class:`dpctl.memory.MemoryUSMDevice`,
            :class:`dpctl.memory.MemoryUSMShared`, or
            :class:`dpctl.memory.MemoryUSMHost`. Recognized strings are
            ``"device"``, ``"shared"``, or ``"host"``. Additional arguments to
            the USM memory alloctors can be passed in a dictionary specified
            via ``buffer_ctor_kwrds`` keyword parameter.
            Default: ``"device"``.
        offset (int, optional):
            Offset of the array element with all zero indexes relative to the
            start of the provided `buffer` in elements. The argument is ignored
            if the ``buffer`` value is a string and the memory is allocated by
            the constructor. Default: ``0``.
        order ({"C", "F"}, optional):
            The memory layout of the array when constructing using a new
            allocation. Value ``"C"`` corresponds to C-contiguous, or row-major
            memory layout, while value ``"F"`` corresponds to F-contiguous, or
            column-major layout. Default: ``"C"``.
        buffer_ctor_kwargs (dict, optional):
            Dictionary with keyword parameters to use when creating a new USM
            memory allocation. See :class:`dpctl.memory.MemoryUSMShared` for
            supported keyword arguments.
        array_namespace (module, optional):
            Array namespace module associated with this array.
            Default: ``None``.

    ``buffer`` can be ``"shared"``, ``"host"``, ``"device"`` to allocate
    new device memory by calling respective constructor with
    the specified ``buffer_ctor_kwrds``; ``buffer`` can be an
    instance of :class:`dpctl.memory.MemoryUSMShared`,
    :class:`dpctl.memory.MemoryUSMDevice`, or
    :class:`dpctl.memory.MemoryUSMHost`; ``buffer`` can also be
    another :class:`dpctl.tensor.usm_ndarray` instance, in which case its
    underlying ``MemoryUSM*`` buffer is used.
    """

    cdef void _reset(usm_ndarray self):
        """
        Initializes member fields
        """
        self.base_ = None
        self.array_namespace_ = None
        self.nd_ = -1
        self.data_ = <char *>0
        self.shape_ = <Py_ssize_t *>0
        self.strides_ = <Py_ssize_t *>0
        self.flags_ = 0

    cdef void _cleanup(usm_ndarray self):
        if (self.shape_):
            PyMem_Free(self.shape_)
        if (self.strides_):
            PyMem_Free(self.strides_)
        self._reset()

    cdef usm_ndarray _clone(usm_ndarray self):
        """
        Provides a copy of Python object pointing to the same data
        """
        cdef Py_ssize_t offset_elems = self.get_offset()
        cdef usm_ndarray res = usm_ndarray.__new__(
            usm_ndarray, _make_int_tuple(self.nd_, self.shape_),
            dtype=_make_typestr(self.typenum_),
            strides=(
                _make_int_tuple(self.nd_, self.strides_) if (self.strides_)
                else None),
            buffer=self.base_,
            offset=offset_elems,
            order=('C' if (self.flags_ & USM_ARRAY_C_CONTIGUOUS) else 'F')
        )
        res.flags_ = self.flags_
        res.array_namespace_ = self.array_namespace_
        if (res.data_ != self.data_):
            raise InternalUSMArrayError(
                "Data pointers of cloned and original objects are different.")
        return res

    def __cinit__(self, shape, dtype="|f8", strides=None, buffer='device',
                  Py_ssize_t offset=0, order='C',
                  buffer_ctor_kwargs=dict(),
                  array_namespace=None):
        """
        strides and offset must be given in units of array elements.
        buffer can be strings ('device'|'shared'|'host' to allocate new memory)
        or dpctl.memory.MemoryUSM* buffers, or usm_ndrray instances.
        """
        cdef int nd = 0
        cdef int typenum = 0
        cdef int itemsize = 0
        cdef int err = 0
        cdef int contig_flag = 0
        cdef Py_ssize_t *shape_ptr = NULL
        cdef Py_ssize_t ary_nelems = 0
        cdef Py_ssize_t ary_nbytes = 0
        cdef Py_ssize_t *strides_ptr = NULL
        cdef Py_ssize_t _offset = offset
        cdef Py_ssize_t ary_min_displacement = 0
        cdef Py_ssize_t ary_max_displacement = 0
        cdef bint is_fp64 = False
        cdef bint is_fp16 = False

        self._reset()
        if (not isinstance(shape, (list, tuple))
                and not hasattr(shape, 'tolist')):
            try:
                <Py_ssize_t> shape
                shape = [shape, ]
            except Exception:
                raise TypeError("Argument shape must be a list or a tuple.")
        nd = len(shape)
        typenum = dtype_to_typenum(dtype)
        if (typenum < 0):
            if typenum == -2:
                raise ValueError("Data type '" + str(dtype) + "' can only have native byteorder.")
            elif typenum == -1:
                raise ValueError("Data type '" + str(dtype) + "' is not understood.")
            raise TypeError(f"Expected string or a dtype object, got {type(dtype)}")
        itemsize = type_bytesize(typenum)
        if (itemsize < 1):
            raise TypeError("dtype=" + np.dtype(dtype).name + " is not supported.")
        # allocate host C-arrays for shape, strides
        err = _from_input_shape_strides(
            nd, shape, strides, itemsize, <char> ord(order),
            &shape_ptr, &strides_ptr, &ary_nelems,
            &ary_min_displacement, &ary_max_displacement, &contig_flag
        )
        if (err):
            self._cleanup()
            if err == ERROR_MALLOC:
                raise MemoryError("Memory allocation for shape/strides "
                                  "array failed.")
            elif err == ERROR_INCORRECT_ORDER:
                raise ValueError(
                    "Unsupported order='{}' given. "
                    "Supported values are 'C' or 'F'.".format(order))
            elif err == ERROR_UNEXPECTED_STRIDES:
                raise ValueError(
                    "strides={} is not understood".format(strides))
            else:
                raise InternalUSMArrayError(
                    " .. while processing shape and strides.")
        ary_nbytes = (ary_max_displacement -
                      ary_min_displacement + 1) * itemsize
        if isinstance(buffer, dpmem._memory._Memory):
            _buffer = buffer
        elif isinstance(buffer, (str, bytes)):
            if isinstance(buffer, bytes):
                buffer = buffer.decode("UTF-8")
            _offset = -ary_min_displacement
            if (buffer == "shared"):
                _buffer = dpmem.MemoryUSMShared(ary_nbytes,
                                                **buffer_ctor_kwargs)
            elif (buffer == "device"):
                _buffer = dpmem.MemoryUSMDevice(ary_nbytes,
                                                **buffer_ctor_kwargs)
            elif (buffer == "host"):
                _buffer = dpmem.MemoryUSMHost(ary_nbytes,
                                              **buffer_ctor_kwargs)
            else:
                self._cleanup()
                raise ValueError(
                    ("buffer='{}' is not understood. "
                    "Recognized values are 'device', 'shared',  'host', "
                    "an instance of `MemoryUSM*` object, or a usm_ndarray"
                     "").format(buffer))
        elif isinstance(buffer, usm_ndarray):
            _buffer = buffer.usm_data
        else:
            self._cleanup()
            raise ValueError("buffer='{}' was not understood.".format(buffer))
        if (_offset + ary_min_displacement < 0 or
           (_offset + ary_max_displacement + 1) * itemsize > _buffer.nbytes):
            self._cleanup()
            raise ValueError(("buffer='{}' can not accomodate "
                              "the requested array.").format(buffer))
        is_fp64 = (typenum == UAR_DOUBLE or typenum == UAR_CDOUBLE)
        is_fp16 = (typenum == UAR_HALF)
        if (is_fp64 or is_fp16):
            if ((is_fp64 and not _buffer.sycl_device.has_aspect_fp64) or
                (is_fp16 and not _buffer.sycl_device.has_aspect_fp16)
            ):
                raise ValueError(
                    f"Device {_buffer.sycl_device.name} does"
                    f" not support {dtype} natively."
                )
        self.base_ = _buffer
        self.data_ = (<char *> (<size_t> _buffer._pointer)) + itemsize * _offset
        self.shape_ = shape_ptr
        self.strides_ = strides_ptr
        self.typenum_ = typenum
        self.flags_ = (contig_flag | USM_ARRAY_WRITABLE)
        self.nd_ = nd
        self.array_namespace_ = array_namespace

    def __dealloc__(self):
        self._cleanup()

    @property
    def _pointer(self):
        "Returns USM pointer for data allocation encoded as integer"
        return <size_t> self.get_data()

    cdef Py_ssize_t get_offset(self) except *:
        cdef char *mem_ptr = NULL
        cdef char *ary_ptr = self.get_data()
        mem_ptr = <char *>(<size_t> self.base_._pointer)
        byte_offset = ary_ptr - mem_ptr
        item_size = self.get_itemsize()
        if (byte_offset % item_size):
            raise InternalUSMArrayError(
                "byte_offset is not a multiple of item_size.")
        return byte_offset // item_size

    cdef char* get_data(self):
        """Returns the USM pointer for this array."""
        return self.data_

    cdef int get_ndim(self):
        """
        Returns the number of indices needed to address
        an element of this array.
        """
        return self.nd_

    cdef Py_ssize_t* get_shape(self):
        """
        Returns pointer to shape C-array for this array.

        C-array has at least ``ndim`` non-negative elements,
        which determine the range of permissible indices
        addressing individual elements of this array.
        """
        return self.shape_

    cdef Py_ssize_t* get_strides(self):
        """
        Returns pointer to strides C-array for this array.

        The pointer can be NULL (contiguous array), or the
        array size is at least ``ndim`` elements
        """
        return self.strides_

    cdef int get_typenum(self):
        """Returns typenum corresponding to values of this array"""
        return self.typenum_

    cdef int get_itemsize(self):
        """
        Returns itemsize of this arrays in bytes
        """
        return type_bytesize(self.typenum_)

    cdef int get_flags(self):
        """Returns flags of this array"""
        return self.flags_

    cdef object get_base(self):
        """Returns the object owning the USM data addressed by this array"""
        return self.base_

    cdef c_dpctl.SyclQueue get_sycl_queue(self):
        cdef c_dpmem._Memory mem
        if not isinstance(self.base_, dpctl.memory._Memory):
            raise InternalUSMArrayError(
                "This array has unexpected memory owner"
            )
        mem = <c_dpmem._Memory> self.base_
        return mem.queue

    cdef c_dpctl.DPCTLSyclQueueRef get_queue_ref(self) except *:
        """
        Returns a copy of DPCTLSyclQueueRef associated with array
        """
        cdef c_dpctl.SyclQueue q = self.get_sycl_queue()
        cdef c_dpctl.DPCTLSyclQueueRef QRef = q.get_queue_ref()
        cdef c_dpctl.DPCTLSyclQueueRef QRefCopy = NULL
        if QRef is not NULL:
            QRefCopy = c_dpctl.DPCTLQueue_Copy(QRef)
            return QRefCopy
        else:
            raise InternalUSMArrayError(
                "Memory owner of this array is corrupted"
            )

    @property
    def __sycl_usm_array_interface__(self):
        """
        Gives ``__sycl_usm_array_interface__`` dictionary describing
        the array.
        """
        cdef Py_ssize_t byte_offset = -1
        cdef int item_size = -1
        cdef Py_ssize_t elem_offset = -1
        cdef char *mem_ptr = NULL
        cdef char *ary_ptr = NULL
        if (not isinstance(self.base_, dpmem._memory._Memory)):
            raise InternalUSMArrayError(
                "Invalid instance of usm_ndarray ecountered. "
                "Private field base_ has an unexpected type {}.".format(
                    type(self.base_)
                )
            )
        ary_iface = self.base_.__sycl_usm_array_interface__
        mem_ptr = <char *>(<size_t> ary_iface['data'][0])
        ary_ptr = <char *>(<size_t> self.data_)
        ro_flag = False if (self.flags_ & USM_ARRAY_WRITABLE) else True
        ary_iface['data'] = (<size_t> mem_ptr, ro_flag)
        ary_iface['shape'] = self.shape
        if (self.strides_):
            ary_iface['strides'] = _make_int_tuple(self.nd_, self.strides_)
        else:
            if (self.flags_ & USM_ARRAY_C_CONTIGUOUS):
                ary_iface['strides'] = None
            elif (self.flags_ & USM_ARRAY_F_CONTIGUOUS):
                ary_iface['strides'] = _f_contig_strides(self.nd_, self.shape_)
            else:
                raise InternalUSMArrayError(
                    "USM Array is not contiguous and has empty strides"
                )
        ary_iface['typestr'] = _make_typestr(self.typenum_)
        byte_offset = ary_ptr - mem_ptr
        item_size = self.get_itemsize()
        if (byte_offset % item_size):
            raise InternalUSMArrayError(
                "byte_offset is not a multiple of item_size.")
        elem_offset = byte_offset // item_size
        ary_iface['offset'] = elem_offset
        return ary_iface

    @property
    def ndim(self):
        """
        Gives the number of indices needed to address elements of this array.
        """
        return self.nd_

    @property
    def usm_data(self):
        """
        Gives USM memory object underlying usm_array instance.
        """
        return self.get_base()

    @property
    def shape(self):
        """
        Elements of the shape tuple give the lengths of the
        respective array dimensions.
        """
        if self.nd_ > 0:
            return _make_int_tuple(self.nd_, self.shape_)
        else:
            return tuple()

    @shape.setter
    def shape(self, new_shape):
        """
        Setting shape is only allowed when reshaping to the requested
        dimensions can be returned as view. Use :func:`dpctl.tensor.reshape`
        to reshape the array in all other cases.
        """
        cdef int new_nd = -1
        cdef Py_ssize_t nelems = -1
        cdef int err = 0
        cdef Py_ssize_t min_disp = 0
        cdef Py_ssize_t max_disp = 0
        cdef int contig_flag = 0
        cdef Py_ssize_t *shape_ptr = NULL
        cdef Py_ssize_t *strides_ptr = NULL
        cdef Py_ssize_t size = -1
        import operator

        from ._reshape import reshaped_strides

        new_nd = len(new_shape)
        try:
            new_shape = tuple(operator.index(dim) for dim in new_shape)
        except TypeError:
            raise TypeError(
                "Target shape must be a finite iterable of integers"
            )
        size = shape_to_elem_count(self.nd_, self.shape_)
        if not np.prod(new_shape) == size:
            raise TypeError(
                f"Can not reshape array of size {self.size} into {new_shape}"
            )
        if size > 0:
            new_strides = reshaped_strides(
               self.shape,
               self.strides,
               new_shape
            )
        else:
            new_strides = (1,) * len(new_shape)
        if new_strides is None:
            raise AttributeError(
                "Incompatible shape for in-place modification. "
                "Use `reshape()` to make a copy with the desired shape."
            )
        err = _from_input_shape_strides(
            new_nd, new_shape, new_strides,
            self.get_itemsize(),
            b"C",
            &shape_ptr, &strides_ptr,
            &nelems, &min_disp, &max_disp, &contig_flag
        )
        if (err == 0):
            if (self.shape_):
                PyMem_Free(self.shape_)
            if (self.strides_):
                PyMem_Free(self.strides_)
            self.flags_ = contig_flag
            self.nd_ = new_nd
            self.shape_ = shape_ptr
            self.strides_ = strides_ptr
        else:
            raise InternalUSMArrayError(
                "Encountered in shape setter, error code {err}".format(err)
            )

    @property
    def strides(self):
        """
        Returns memory displacement in array elements, upon unit
        change of respective index.

        E.g. for strides (s1, s2, s3) and multi-index (i1, i2, i3)

           a[i1, i2, i3] == (&a[0,0,0])[ s1*s1 + s2*i2 + s3*i3]
        """
        if (self.strides_):
            return _make_int_tuple(self.nd_, self.strides_)
        else:
            if (self.flags_ & USM_ARRAY_C_CONTIGUOUS):
                return _c_contig_strides(self.nd_, self.shape_)
            elif (self.flags_ & USM_ARRAY_F_CONTIGUOUS):
                return _f_contig_strides(self.nd_, self.shape_)
            else:
                raise ValueError("Inconsitent usm_ndarray data")

    @property
    def flags(self):
        """
        Returns :class:`dpctl.tensor._flags` object.
        """
        return _flags.Flags(self, self.flags_)

    cdef _set_writable_flag(self, int flag):
        cdef int arr_fl = self.flags_
        arr_fl ^= (arr_fl & USM_ARRAY_WRITABLE)  # unset WRITABLE flag
        arr_fl |= (USM_ARRAY_WRITABLE if flag else 0)
        self.flags_ = arr_fl

    @property
    def usm_type(self):
        """
        USM type of underlying memory. Can be ``"device"``, ``"shared"``,
        or ``"host"``.

        See: https://docs.oneapi.com/versions/latest/dpcpp/iface/usm.html
        """
        return self.base_.get_usm_type()

    @property
    def itemsize(self):
        """
        Size of array element in bytes.
        """
        return self.get_itemsize()

    @property
    def nbytes(self):
        """
        Total bytes consumed by the elements of the array.
        """
        return (
            shape_to_elem_count(self.nd_, self.shape_) *
            self.get_itemsize())

    @property
    def size(self):
        """
        Number of elements in the array.
        """
        return shape_to_elem_count(self.nd_, self.shape_)

    @property
    def dtype(self):
        """
        Returns NumPy's dtype corresponding to the type of the array elements.
        """
        return np.dtype(_make_typestr(self.typenum_))

    @property
    def sycl_queue(self):
        """
        Returns :class:`dpctl.SyclQueue` object associated with USM data.
        """
        return self.get_sycl_queue()

    @property
    def sycl_device(self):
        """
        Returns :class:`dpctl.SyclDevice` object on which USM data was allocated.
        """
        q = self.sycl_queue
        return q.sycl_device

    @property
    def device(self):
        """
        Returns data-API object representing residence of the array data.
        """
        return Device.create_device(self.sycl_queue)

    @property
    def sycl_context(self):
        """
        Returns :class:`dpctl.SyclContext` object to which USM data is bound.
        """
        q = self.sycl_queue
        return q.sycl_context

    @property
    def T(self):
        """ Returns tranposed array for 2D array, raises `ValueError`
        otherwise.
        """
        if self.nd_ == 2:
            return _transpose(self)
        else:
            raise ValueError(
                "array.T requires array to have 2 dimensions. "
                "Use array.mT to transpose stacks of matrices and "
                "dpctl.tensor.permute_dims() to permute dimensions."
            )

    @property
    def mT(self):
        """ Returns array where the last two dimensions are transposed.
        """
        if self.nd_ < 2:
            raise ValueError(
                "array.mT requires array to have at least 2-dimensons."
            )
        return _m_transpose(self)

    @property
    def real(self):
        """ Returns real component for arrays with complex data-types
        and returns itself for all other data-types.
        """
        if (self.typenum_ < UAR_CFLOAT):
            # elements are real
            return self
        if (self.typenum_ < UAR_TYPE_SENTINEL):
            return _real_view(self)

    @property
    def imag(self):
        """ Returns imaginary component for arrays with complex data-types
        and returns zero array for all other data-types.
        """
        if (self.typenum_ < UAR_CFLOAT):
            # elements are real
            return _zero_like(self)
        if (self.typenum_ < UAR_TYPE_SENTINEL):
            return _imag_view(self)

    def __getitem__(self, ind):
        cdef tuple _meta = _basic_slice_meta(
            ind, (<object>self).shape, (<object> self).strides,
            self.get_offset())
        cdef usm_ndarray res

        if len(_meta) < 5:
            raise RuntimeError

        res = usm_ndarray.__new__(
            usm_ndarray,
            _meta[0],
            dtype=_make_typestr(self.typenum_),
            strides=_meta[1],
            buffer=self.base_,
            offset=_meta[2]
        )
        res.flags_ |= (self.flags_ & USM_ARRAY_WRITABLE)
        res.array_namespace_ = self.array_namespace_

        adv_ind = _meta[3]
        adv_ind_start_p = _meta[4]

        if adv_ind_start_p < 0:
            return res

        from ._copy_utils import _extract_impl, _nonzero_impl, _take_multi_index
        if len(adv_ind) == 1 and adv_ind[0].dtype == dpt_bool:
            return _extract_impl(res, adv_ind[0], axis=adv_ind_start_p)

        if any(ind.dtype == dpt_bool for ind in adv_ind):
            adv_ind_int = list()
            for ind in adv_ind:
                if ind.dtype == dpt_bool:
                    adv_ind_int.extend(_nonzero_impl(ind))
                else:
                    adv_ind_int.append(ind)
            return _take_multi_index(res, tuple(adv_ind_int), adv_ind_start_p)

        return _take_multi_index(res, adv_ind, adv_ind_start_p)


    def to_device(self, target):
        """ to_device(target_device)

        Transfers this array to specified target device.

        :Example:
            .. code-block:: python

                import dpctl
                import dpctl.tensor as dpt

                x = dpt.full(10**6, 2, dtype="int64")
                q_prof = dpctl.SyclQueue(
                    x.sycl_device, property="enable_profiling")
                # return a view with profile-enabled queue
                y = x.to_device(q_prof)
                timer = dpctl.SyclTimer()
                with timer(q_prof):
                    z = y * y
                print(timer.dt)

        Args:
            target_device (object):
                Array API concept of target device.
                It can be a oneAPI filter selector string,
                an instance of :class:`dpctl.SyclDevice` corresponding to a
                non-partitioned SYCL device, an instance of
                :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device`
                object returned by :attr:`dpctl.tensor.usm_array.device`.

        Returns:
            usm_ndarray:
                A view if data copy is not required, and a copy otherwise.
                If copying is required, it is done by copying from the original
                allocation device to the host, followed by copying from host
                to the target device.
        """
        cdef c_dpctl.DPCTLSyclQueueRef QRef = NULL
        cdef c_dpmem._Memory arr_buf
        d = Device.create_device(target)
        if (d.sycl_context == self.sycl_context):
            arr_buf = <c_dpmem._Memory> self.usm_data
            QRef = (<c_dpctl.SyclQueue> d.sycl_queue).get_queue_ref()
            view_buffer = c_dpmem._Memory.create_from_usm_pointer_size_qref(
                arr_buf.memory_ptr,
                arr_buf.nbytes,
                QRef,
                memory_owner = arr_buf
            )
            res = usm_ndarray(
                self.shape,
                self.dtype,
                buffer=view_buffer,
                strides=self.strides,
                offset=self.get_offset()
            )
            res.flags_ = self.flags.flags
            return res
        else:
            nbytes = self.usm_data.nbytes
            copy_buffer = type(self.usm_data)(
                nbytes, queue=d.sycl_queue
            )
            copy_buffer.copy_from_device(self.usm_data)
            res = usm_ndarray(
                self.shape,
                self.dtype,
                buffer=copy_buffer,
                strides=self.strides,
                offset=self.get_offset()
            )
            res.flags_ = self.flags.flags
            return res

    def _set_namespace(self, mod):
        """ Sets array namespace to given module `mod`. """
        self.array_namespace_ = mod

    def __array_namespace__(self, api_version=None):
        """
        Returns array namespace, member functions of which
        implement data API.
        """
        return self.array_namespace_

    def __bool__(self):
        if self.size == 1:
            mem_view = dpmem.as_usm_memory(self)
            return mem_view.copy_to_host().view(self.dtype).__bool__()

        if self.size == 0:
            raise ValueError(
                "The truth value of an empty array is ambiguous"
            )

        raise ValueError(
            "The truth value of an array with more than one element is "
            "ambiguous. Use a.any() or a.all()"
        )

    def __float__(self):
        if self.size == 1:
            mem_view = dpmem.as_usm_memory(self)
            return mem_view.copy_to_host().view(self.dtype).__float__()

        raise ValueError(
            "only size-1 arrays can be converted to Python scalars"
        )

    def __complex__(self):
        if self.size == 1:
            mem_view = dpmem.as_usm_memory(self)
            return mem_view.copy_to_host().view(self.dtype).__complex__()

        raise ValueError(
            "only size-1 arrays can be converted to Python scalars"
        )

    def __int__(self):
        if self.size == 1:
            mem_view = dpmem.as_usm_memory(self)
            return mem_view.copy_to_host().view(self.dtype).__int__()

        raise ValueError(
            "only size-1 arrays can be converted to Python scalars"
        )

    def __index__(self):
        if np.issubdtype(self.dtype, np.integer):
            return int(self)

        raise IndexError("only integer arrays are valid indices")

    def __abs__(self):
        return _dispatch_unary_elementwise(self, "abs")

    def __add__(first, other):
        """
        Cython 0.* never calls `__radd__`, always calls `__add__`
        but first argument need not be an instance of this class,
        so dispatching is needed.

        This changes in Cython 3.0, where first is guaranteed to
        be `self`.

        [1] http://docs.cython.org/en/latest/src/userguide/special_methods.html
        """
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "add", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "add", other)
        return NotImplemented

    def __and__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "logical_and", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "logical_and", other)
        return NotImplemented

    def __dlpack__(self, stream=None):
        """
        Produces DLPack capsule.

        Raises:
            MemoryError: when host memory can not be allocated.
            DLPackCreationError: when array is allocated on a partitioned
                SYCL device, or with a non-default context.
            NotImplementedError: when non-default value of `stream` keyword
                is used.
        """
        _caps = c_dlpack.to_dlpack_capsule(self)
        if (stream is None or type(stream) is not dpctl.SyclQueue or
            stream == self.sycl_queue):
            pass
        else:
            ev = self.sycl_queue.submit_barrier()
            stream.submit_barrier(dependent_events=[ev])
        return _caps

    def __dlpack_device__(self):
        """
        Gives a tuple (`device_type`, `device_id`) corresponding to ``DLDevice``
        entry in ``DLTensor`` in DLPack protocol.

        The tuple describes the non-partitioned device where the array
        has been allocated.

        Raises:
            DLPackCreationError: when array is allocation on a partitioned
                SYCL device
        """
        cdef int dev_id = (<c_dpctl.SyclDevice>self.sycl_device).get_overall_ordinal()
        if dev_id < 0:
            raise c_dlpack.DLPackCreationError(
                "DLPack protocol is only supported for non-partitioned devices"
            )
        else:
            return (
                c_dlpack.device_oneAPI,
                dev_id,
            )

    def __eq__(self, other):
        return _dispatch_binary_elementwise(self, "equal", other)

    def __floordiv__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "floor_divide", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "floor_divide", other)
        return NotImplemented

    def __ge__(self, other):
        return _dispatch_binary_elementwise(self, "greater_equal", other)

    def __gt__(self, other):
        return _dispatch_binary_elementwise(self, "greater", other)

    def __invert__(self):
        return _dispatch_unary_elementwise(self, "invert")

    def __le__(self, other):
        return _dispatch_binary_elementwise(self, "less_equal", other)

    def __len__(self):
        if (self.nd_):
            return self.shape[0]
        else:
            raise TypeError("len() of unsized object")

    def __lshift__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "left_shift", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "left_shift", other)
        return NotImplemented

    def __lt__(self, other):
        return _dispatch_binary_elementwise(self, "less", other)

    def __matmul__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "matmul", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "matmul", other)
        return NotImplemented

    def __mod__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "mod", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "mod", other)
        return NotImplemented

    def __mul__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "multiply", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "multiply", other)
        return NotImplemented

    def __ne__(self, other):
        return _dispatch_binary_elementwise(self, "not_equal", other)

    def __neg__(self):
        return _dispatch_unary_elementwise(self, "negative")

    def __or__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "logical_or", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "logical_or", other)
        return NotImplemented

    def __pos__(self):
        return self  # _dispatch_unary_elementwise(self, "positive")

    def __pow__(first, other, mod):
        "See comment in __add__"
        if mod is None:
            if isinstance(first, usm_ndarray):
                return _dispatch_binary_elementwise(first, "power", other)
            elif isinstance(other, usm_ndarray):
                return _dispatch_binary_elementwise(first, "power", other)
        return NotImplemented

    def __rshift__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "right_shift", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "right_shift", other)
        return NotImplemented

    def __setitem__(self, key, rhs):
        cdef tuple _meta
        cdef usm_ndarray Xv

        if (self.flags_ & USM_ARRAY_WRITABLE) == 0:
            raise ValueError("Can not modify read-only array.")

        _meta = _basic_slice_meta(
            key, (<object>self).shape, (<object> self).strides,
            self.get_offset()
        )

        if len(_meta) < 5:
            raise RuntimeError

        Xv = usm_ndarray.__new__(
            usm_ndarray,
            _meta[0],
            dtype=_make_typestr(self.typenum_),
            strides=_meta[1],
            buffer=self.base_,
            offset=_meta[2],
        )
        # set flags and namespace
        Xv.flags_ |= (self.flags_ & USM_ARRAY_WRITABLE)
        Xv.array_namespace_ = self.array_namespace_

        from ._copy_utils import (
            _copy_from_numpy_into,
            _copy_from_usm_ndarray_to_usm_ndarray,
            _nonzero_impl,
            _place_impl,
            _put_multi_index,
        )

        adv_ind = _meta[3]
        adv_ind_start_p = _meta[4]

        if adv_ind_start_p < 0:
            # basic slicing
            if isinstance(rhs, usm_ndarray):
                _copy_from_usm_ndarray_to_usm_ndarray(Xv, rhs)
            else:
                if hasattr(rhs, "__sycl_usm_array_interface__"):
                    from dpctl.tensor import asarray
                    try:
                        rhs_ar = asarray(rhs)
                        _copy_from_usm_ndarray_to_usm_ndarray(Xv, rhs_ar)
                    except Exception:
                        raise ValueError(
                            f"Input of type {type(rhs)} could not be "
                            "converted to usm_ndarray"
                        )
                else:
                    rhs_np = np.asarray(rhs)
                    if type_bytesize(rhs_np.dtype.num) < 0:
                        raise ValueError(
                            f"Input of type {type(rhs)} can not be "
                            "assigned to usm_ndarray because of "
                            f"unsupported data type '{rhs_np.dtype}'"
                        )
                    try:
                        _copy_from_numpy_into(Xv, rhs_np)
                    except Exception:
                        raise ValueError(
                            f"Input of type {type(rhs)} could not be "
                            "copied into dpctl.tensor.usm_ndarray"
                        )
            return

        if len(adv_ind) == 1 and adv_ind[0].dtype == dpt_bool:
            _place_impl(Xv, adv_ind[0], rhs, axis=adv_ind_start_p)
            return

        if any(ind.dtype == dpt_bool for ind in adv_ind):
            adv_ind_int = list()
            for ind in adv_ind:
                if ind.dtype == dpt_bool:
                    adv_ind_int.extend(_nonzero_impl(ind))
                else:
                    adv_ind_int.append(ind)
            _put_multi_index(Xv, tuple(adv_ind_int), adv_ind_start_p, rhs)
            return

        _put_multi_index(Xv, adv_ind, adv_ind_start_p, rhs)
        return


    def __sub__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "subtract", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "subtract", other)
        return NotImplemented

    def __truediv__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "true_divide", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "true_divide", other)
        return NotImplemented

    def __xor__(first, other):
        "See comment in __add__"
        if isinstance(first, usm_ndarray):
            return _dispatch_binary_elementwise(first, "logical_xor", other)
        elif isinstance(other, usm_ndarray):
            return _dispatch_binary_elementwise2(first, "logical_xor", other)
        return NotImplemented

    def __radd__(self, other):
        return _dispatch_binary_elementwise(self, "add", other)

    def __rand__(self, other):
        return _dispatch_binary_elementwise(self, "logical_and", other)

    def __rfloordiv__(self, other):
        return _dispatch_binary_elementwise2(other, "floor_divide", self)

    def __rlshift__(self, other):
        return _dispatch_binary_elementwise2(other, "left_shift", self)

    def __rmatmul__(self, other):
        return _dispatch_binary_elementwise2(other, "matmul", self)

    def __rmod__(self, other):
        return _dispatch_binary_elementwise2(other, "mod", self)

    def __rmul__(self, other):
        return _dispatch_binary_elementwise(self, "multiply", other)

    def __ror__(self, other):
        return _dispatch_binary_elementwise(self, "logical_or", other)

    def __rpow__(self, other):
        return _dispatch_binary_elementwise2(other, "power", self)

    def __rrshift__(self, other):
        return _dispatch_binary_elementwise2(other, "right_shift", self)

    def __rsub__(self, other):
        return _dispatch_binary_elementwise2(other, "subtract", self)

    def __rtruediv__(self, other):
        return _dispatch_binary_elementwise2(other, "true_divide", self)

    def __rxor__(self, other):
        return _dispatch_binary_elementwise2(other, "logical_xor", self)

    def __iadd__(self, other):
        res = self.__add__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __iand__(self, other):
        res = self.__and__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __ifloordiv__(self, other):
        res = self.__floordiv__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __ilshift__(self, other):
        res = self.__lshift__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __imatmul__(self, other):
        res = self.__matmul__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __imod__(self, other):
        res = self.__mod__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __imul__(self, other):
        res = self.__mul__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __ior__(self, other):
        res = self.__or__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __ipow__(self, other):
        res = self.__pow__(other, None)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __irshift__(self, other):
        res = self.__rshift__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __isub__(self, other):
        res = self.__sub__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __itruediv__(self, other):
        res = self.__truediv__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __ixor__(self, other):
        res = self.__xor__(other)
        if res is NotImplemented:
            return res
        self.__setitem__(Ellipsis, res)
        return self

    def __str__(self):
        return usm_ndarray_str(self)

    def __repr__(self):
        return usm_ndarray_repr(self)


cdef usm_ndarray _real_view(usm_ndarray ary):
    """
    View into real parts of a complex type array
    """
    cdef int r_typenum_ = -1
    cdef usm_ndarray r = None
    cdef Py_ssize_t offset_elems = 0

    if (ary.typenum_ == UAR_CFLOAT):
        r_typenum_ = UAR_FLOAT
    elif (ary.typenum_ == UAR_CDOUBLE):
        r_typenum_ = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_real_view call on array of non-complex type.")

    offset_elems = ary.get_offset() * 2
    r = usm_ndarray.__new__(
        usm_ndarray,
        _make_int_tuple(ary.nd_, ary.shape_),
        dtype=_make_typestr(r_typenum_),
        strides=tuple(2 * si for si in ary.strides),
        buffer=ary.base_,
        offset=offset_elems,
        order=('C' if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else 'F')
    )
    r.flags_ = ary.flags_
    r.array_namespace_ = ary.array_namespace_
    return r


cdef usm_ndarray _imag_view(usm_ndarray ary):
    """
    View into imaginary parts of a complex type array
    """
    cdef int r_typenum_ = -1
    cdef usm_ndarray r = None
    cdef Py_ssize_t offset_elems = 0

    if (ary.typenum_ == UAR_CFLOAT):
        r_typenum_ = UAR_FLOAT
    elif (ary.typenum_ == UAR_CDOUBLE):
        r_typenum_ = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_imag_view call on array of non-complex type.")

    # displace pointer to imaginary part
    offset_elems = 2 * ary.get_offset() + 1
    r = usm_ndarray.__new__(
        usm_ndarray,
        _make_int_tuple(ary.nd_, ary.shape_),
        dtype=_make_typestr(r_typenum_),
        strides=tuple(2 * si for si in ary.strides),
        buffer=ary.base_,
        offset=offset_elems,
        order=('C' if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else 'F')
    )
    r.flags_ = ary.flags_
    r.array_namespace_ = ary.array_namespace_
    return r


cdef usm_ndarray _transpose(usm_ndarray ary):
    """
    Construct transposed array without copying the data
    """
    cdef usm_ndarray r = usm_ndarray.__new__(
        usm_ndarray,
        _make_reversed_int_tuple(ary.nd_, ary.shape_),
        dtype=_make_typestr(ary.typenum_),
        strides=(
            _make_reversed_int_tuple(ary.nd_, ary.strides_)
            if (ary.strides_) else None),
        buffer=ary.base_,
        order=('F' if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else 'C'),
        offset=ary.get_offset()
    )
    r.flags_ |= (ary.flags_ & USM_ARRAY_WRITABLE)
    return r


cdef usm_ndarray _m_transpose(usm_ndarray ary):
    """
    Construct matrix transposed array
    """
    cdef usm_ndarray r = usm_ndarray.__new__(
        usm_ndarray,
        _swap_last_two(_make_int_tuple(ary.nd_, ary.shape_)),
        dtype=_make_typestr(ary.typenum_),
        strides=_swap_last_two(ary.strides),
        buffer=ary.base_,
        order=('F' if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else 'C'),
        offset=ary.get_offset()
    )
    r.flags_ |= (ary.flags_ & USM_ARRAY_WRITABLE)
    return r


cdef usm_ndarray _zero_like(usm_ndarray ary):
    """
    Make C-contiguous array of zero elements with same shape
    and type as ary.
    """
    cdef dt = _make_typestr(ary.typenum_)
    cdef usm_ndarray r = usm_ndarray(
        _make_int_tuple(ary.nd_, ary.shape_),
        dtype=dt,
        buffer=ary.base_.get_usm_type()
    )
    r.base_.memset()
    return r


cdef api char* UsmNDArray_GetData(usm_ndarray arr):
    """Get allocation pointer of zero index element of array """
    return arr.get_data()


cdef api int UsmNDArray_GetNDim(usm_ndarray arr):
    """Get array rank: length of its shape"""
    return arr.get_ndim()


cdef api Py_ssize_t* UsmNDArray_GetShape(usm_ndarray arr):
    """Get host pointer to shape vector"""
    return arr.get_shape()


cdef api Py_ssize_t* UsmNDArray_GetStrides(usm_ndarray arr):
    """Get host pointer to strides vector"""
    return arr.get_strides()


cdef api int UsmNDArray_GetTypenum(usm_ndarray arr):
    """Get type number for data type of array elements"""
    return arr.get_typenum()


cdef api int UsmNDArray_GetElementSize(usm_ndarray arr):
    """Get array element size in bytes"""
    return arr.get_itemsize()


cdef api int UsmNDArray_GetFlags(usm_ndarray arr):
    """Get flags of array"""
    return arr.get_flags()


cdef api c_dpctl.DPCTLSyclQueueRef UsmNDArray_GetQueueRef(usm_ndarray arr):
    """Get DPCTLSyclQueueRef for queue associated with the array"""
    return arr.get_queue_ref()


cdef api Py_ssize_t UsmNDArray_GetOffset(usm_ndarray arr):
    """Get offset of zero-index array element from the beginning of the USM
    allocation"""
    return arr.get_offset()


cdef api void UsmNDArray_SetWritableFlag(usm_ndarray arr, int flag):
    """Set/unset USM_ARRAY_WRITABLE in the given array `arr`."""
    arr._set_writable_flag(flag)

cdef api object UsmNDArray_MakeSimpleFromMemory(
    int nd, const Py_ssize_t *shape, int typenum,
    c_dpmem._Memory mobj, Py_ssize_t offset, char order
):
    """Create contiguous usm_ndarray.

    Args:
        nd: number of dimensions (non-negative)
        shape: array of nd non-negative array's sizes along each dimension
        typenum: array elemental type number
        ptr: pointer to the start of allocation
        QRef: DPCTLSyclQueueRef associated with the allocation
        offset: distance between element with zero multi-index and the
                start of allocation
        oder: Memory layout of the array. Use 'C' for C-contiguous or
              row-major layout; 'F' for F-contiguous or column-major layout
    Returns:
        Created usm_ndarray instance
    """
    cdef object shape_tuple = _make_int_tuple(nd, <Py_ssize_t *>shape)
    cdef usm_ndarray arr = usm_ndarray(
        shape_tuple,
        dtype=_make_typestr(typenum),
        buffer=mobj,
        offset=offset,
        order=<bytes>(order)
    )
    return arr


cdef api object UsmNDArray_MakeSimpleFromPtr(
    size_t nelems,
    int typenum,
    c_dpctl.DPCTLSyclUSMRef ptr,
    c_dpctl.DPCTLSyclQueueRef QRef,
    object owner
):
    """Create 1D contiguous usm_ndarray from pointer.

    Args:
        nelems: number of elements in array
        typenum: array elemental type number
        ptr: pointer to the start of allocation
        QRef: DPCTLSyclQueueRef associated with the allocation
        owner: Python object managing lifetime of USM allocation.
               Value None implies transfer of USM allocation ownership
               to the created array object.
    Returns:
        Created usm_ndarray instance
    """
    cdef size_t itemsize = type_bytesize(typenum)
    cdef size_t nbytes = itemsize * nelems
    cdef c_dpmem._Memory mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
        ptr, nbytes, QRef, memory_owner=owner
    )
    cdef usm_ndarray arr = usm_ndarray(
        (nelems,),
        dtype=_make_typestr(typenum),
        buffer=mobj
    )
    return arr

cdef api object UsmNDArray_MakeFromPtr(
    int nd,
    const Py_ssize_t *shape,
    int typenum,
    const Py_ssize_t *strides,
    c_dpctl.DPCTLSyclUSMRef ptr,
    c_dpctl.DPCTLSyclQueueRef QRef,
    Py_ssize_t offset,
    object owner
):
    """
    General usm_ndarray constructor from externally made USM-allocation.

    Args:
        nd: number of dimensions (non-negative)
        shape: array of nd non-negative array's sizes along each dimension
        typenum: array elemental type number
        strides: array of nd strides along each dimension in elements
        ptr: pointer to the start of allocation
        QRef: DPCTLSyclQueueRef associated with the allocation
        offset: distance between element with zero multi-index and the
                start of allocation
        owner: Python object managing lifetime of USM allocation.
               Value None implies transfer of USM allocation ownership
               to the created array object.
    Returns:
        Created usm_ndarray instance
    """
    cdef size_t itemsize = type_bytesize(typenum)
    cdef int err = 0
    cdef size_t nelems = 1
    cdef Py_ssize_t min_disp = 0
    cdef Py_ssize_t max_disp = 0
    cdef Py_ssize_t step_ = 0
    cdef Py_ssize_t dim_ = 0
    cdef it = 0
    cdef c_dpmem._Memory mobj
    cdef usm_ndarray arr
    cdef object obj_shape
    cdef object obj_strides

    if (nd < 0):
        raise ValueError("Dimensionality must be non-negative")
    if (ptr is NULL or QRef is NULL):
        raise ValueError(
            "Non-null USM allocation pointer and QRef are expected"
        )
    if (nd == 0):
        # case of 0d scalars
        mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
            ptr, itemsize, QRef, memory_owner=owner
        )
        arr = usm_ndarray(
            tuple(),
            dtype=_make_typestr(typenum),
            buffer=mobj
        )
        return arr
    if (shape is NULL or strides is NULL):
        raise ValueError("Both shape and stride vectors are required")
    for it in range(nd):
        dim_ = shape[it]
        if dim_ < 0:
            raise ValueError(
                f"Dimension along axis {it} must be non-negative"
            )
        nelems *= dim_
        if dim_ > 0:
            step_ = strides[it]
            if step_ > 0:
                max_disp += step_ * (dim_ - 1)
            else:
                min_disp += step_ * (dim_ - 1)

    obj_shape = _make_int_tuple(nd, shape)
    obj_strides = _make_int_tuple(nd, strides)
    if nelems == 0:
        mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
            ptr, itemsize, QRef, memory_owner=owner
        )
        arr = usm_ndarray(
            obj_shape,
            dtype=_make_typestr(typenum),
            strides=obj_strides,
            buffer=mobj,
            offset=0
        )
        return arr
    if offset + min_disp < 0:
        raise ValueError(
            "Given shape, strides and offset reference out-of-bound memory"
        )
    nbytes = itemsize * (offset + max_disp + 1)
    mobj = c_dpmem._Memory.create_from_usm_pointer_size_qref(
        ptr, nbytes, QRef, memory_owner=owner
    )
    arr = usm_ndarray(
        obj_shape,
        dtype=_make_typestr(typenum),
        strides=obj_strides,
        buffer=mobj,
        offset=offset
    )
    return arr


def _is_object_with_buffer_protocol(o):
   "Returns True if object support Python buffer protocol"
   return _is_buffer(o)
