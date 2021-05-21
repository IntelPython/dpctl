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

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

import numpy as np

import dpctl
import dpctl.memory as dpmem

from ._device import Device

from cpython.mem cimport PyMem_Free
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem

cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpmem


cdef extern from "usm_array.hpp" namespace "usm_array":
    cdef cppclass usm_array:
        usm_array(char *, int, size_t*, Py_ssize_t *,
                  int, int, c_dpctl.DPCTLSyclQueueRef) except +


include "_stride_utils.pxi"
include "_types.pxi"
include "_slicing.pxi"

cdef class InternalUSMArrayError(Exception):
    """
    A InternalError exception is raised when internal
    inconsistency has been detected.
    """
    pass


cdef class usm_ndarray:
    """
    usm_ndarray(
        shape, dtype="|f8", strides=None, buffer='device',
        offset=0, order='C',
        buffer_ctor_kwargs=dict()
    )

    See :class:`dpctl.memory.MemoryUSMShared` for allowed
    keyword arguments.

    `buffer` can be 'shared', 'host', 'device' to allocate
    new device memory by calling respective constructor with
    the specified `buffer_ctor_kwrds`; `buffer` can be an
    instance of :class:`dpctl.memory.MemoryUSMShared`,
    :class:`dpctl.memory.MemoryUSMDevice`, or
    :class:`dpctl.memory.MemoryUSMHost`; `buffer` can also be
    another usm_ndarray instance, in which case its underlying
    MemoryUSM* buffer is used for buffer.
    """

    cdef void _reset(usm_ndarray self):
        """
        Initializes member fields
        """
        self.base_ = None
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
        cdef int item_size = self.get_itemsize()
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
        if (res.data_ != self.data_):
            raise InternalUSMArrayError(
                "Data pointers of cloned and original objects are different.")
        return res

    def __cinit__(self, shape, dtype="|f8", strides=None, buffer='device',
                  Py_ssize_t offset=0, order='C', buffer_ctor_kwargs=dict()):
        """
        strides and offset must be given in units of array elements.
        buffer can be strings ('device'|'shared'|'host' to allocate new memory)
        or dpctl.memory.MemoryUSM* buffers, or usm_ndrray instances.
        """
        cdef int nd = 9
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
        cdef char * data_ptr = NULL

        self._reset()
        if (not isinstance(shape, (list, tuple))
                and not hasattr(shape, 'tolist')):
            raise TypeError("Argument shape must be a list of a tuple.")
        nd = len(shape)
        typenum = dtype_to_typenum(dtype)
        itemsize = type_bytesize(typenum)
        if (itemsize < 1):
            raise TypeError("dtype=" + dtype + " is not supported.")
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
                    "buffer='{}' is not understood. "
                    "Recognized values are 'device', 'shared',  'host', "
                    "an instance of `MemoryUSM*` object, or a usm_ndarray"
                    "".format(buffer))
        elif isinstance(buffer, usm_ndarray):
            _buffer = buffer.usm_data
        else:
            self._cleanup()
            raise ValueError("buffer='{}' was not understood.".format(buffer))
        if (_offset + ary_min_displacement < 0 or
           (_offset + ary_max_displacement + 1) * itemsize > _buffer.nbytes):
            self._cleanup()
            raise ValueError("buffer='{}' can not accomodate the requested "
                             "array.".format(buffer))
        self.base_ = _buffer
        self.data_ = (<char *> (<size_t> _buffer._pointer)) + itemsize * _offset
        self.shape_ = shape_ptr
        self.strides_ = strides_ptr
        self.typenum_ = typenum
        self.flags_ = contig_flag
        self.nd_ = nd

    def __dealloc__(self):
        self._cleanup()

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

        C-array has at least `ndim` non-negative elements,
        which determine the range of permissible indices
        addressing individual elements of this array.
        """
        return self.shape_

    cdef Py_ssize_t* get_strides(self):
        """
        Returns pointer to strides C-array for this array.

        The pointer can be NULL (contiguous array), or the
        array size is at least `ndim` elements
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
        Gives __sycl_usm_array_interface__ dictionary describing the array
        """
        cdef Py_ssize_t byte_offset = -1
        cdef int item_size = -1
        cdef Py_ssize_t elem_offset = -1
        cdef char *mem_ptr = NULL
        cdef char *ary_ptr = NULL
        if (not isinstance(self.base_, dpmem._memory._Memory)):
            raise ValueError("Invalid instance of usm_ndarray ecountered")
        ary_iface = self.base_.__sycl_usm_array_interface__
        mem_ptr = <char *>(<size_t> ary_iface['data'][0])
        ary_ptr = <char *>(<size_t> self.data_)
        ro_flag = False if (self.flags_ & USM_ARRAY_WRITEABLE) else True
        ary_iface['data'] = (<size_t> ary_ptr, ro_flag)
        ary_iface['shape'] = _make_int_tuple(self.nd_, self.shape_)
        if (self.strides_):
            ary_iface['strides'] = _make_int_tuple(self.nd_, self.strides_)
        else:
            if (self.flags_ & USM_ARRAY_C_CONTIGUOUS):
                ary_iface['strides'] = None
            elif (self.flags_ & USM_ARRAY_F_CONTIGUOUS):
                ary_iface['strides'] = _f_contig_strides(self.nd_, self.shape_)
            else:
                raise ValueError("USM Array is not contiguous and "
                                 "has empty strides")
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
        return int(self.nd_)

    @property
    def usm_data(self):
        """
        Gives USM memory object underlying usm_array instance.
        """
        return self.base_

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
        Currently returns integer whose bits correspond to the flags.
        """
        return self.flags_

    @property
    def usm_type(self):
        """
        USM type of underlying memory. Can be 'device', 'shared', or 'host'.

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
        Returns `dpctl.SyclQueue` object associated with USM data.
        """
        return self.get_sycl_queue()

    @property
    def sycl_device(self):
        """
        Returns `dpctl.SyclDevice` object on which USM data was allocated.
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
        Returns `dpctl.SyclContext` object to which USM data is bound.
        """
        q = self.sycl_queue
        return q.sycl_context

    @property
    def T(self):
        if self.nd_ < 2:
            return self
        else:
            return _transpose(self)

    @property
    def real(self):
        if (self.typenum_ < UAR_CFLOAT):
            # elements are real
            return self
        if (self.typenum_ < UAR_TYPE_SENTINEL):
            return _real_view(self)

    @property
    def imag(self):
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

        res = usm_ndarray.__new__(
            usm_ndarray,
            _meta[0],
            dtype=_make_typestr(self.typenum_),
            strides=_meta[1],
            buffer=self.base_,
            offset=_meta[2]
        )
        res.flags_ |= (self.flags_ & USM_ARRAY_WRITEABLE)
        return res

    def to_device(self, target_device):
        """
        Transfer array to target device
        """
        d = Device.create_device(target_device)
        if (d.sycl_device == self.sycl_device):
            return self
        elif (d.sycl_context == self.sycl_context):
            res = usm_ndarray(
                self.shape,
                self.dtype,
                buffer=self.usm_data,
                strides=self.strides,
                offset=self.get_offset()
            )
            res.flags_ = self.flags
            return res
        else:
            nbytes = self.usm_data.nbytes
            new_buffer = type(self.usm_data)(
                nbytes, queue=d.sycl_queue
            )
            new_buffer.copy_from_device(self.usm_data)
            res = usm_ndarray(
                self.shape,
                self.dtype,
                buffer=new_buffer,
                strides=self.strides,
                offset=self.get_offset()
            )
            res.flags_ = self.flags
            return res


cdef usm_ndarray _real_view(usm_ndarray ary):
    """
    View into real parts of a complex type array
    """
    cdef usm_ndarray r = ary._clone()
    if (ary.typenum_ == UAR_CFLOAT):
        r.typenum_ = UAR_FLOAT
    elif (ary.typenum_ == UAR_CDOUBLE):
        r.typenum_ = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_real_view call on array of non-complex type.")
    return r


cdef usm_ndarray _imag_view(usm_ndarray ary):
    """
    View into imaginary parts of a complex type array
    """
    cdef usm_ndarray r = ary._clone()
    if (ary.typenum_ == UAR_CFLOAT):
        r.typenum_ = UAR_FLOAT
    elif (ary.typenum_ == UAR_CDOUBLE):
        r.typenum_ = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_real_view call on array of non-complex type.")
    # displace pointer to imaginary part
    r.data_ = r.data_ + type_bytesize(r.typenum_)
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
        order=('F' if (ary.flags_ & USM_ARRAY_C_CONTIGUOUS) else 'C')
    )
    r.flags_ |= (ary.flags_ & USM_ARRAY_WRITEABLE)
    return r


cdef usm_ndarray _zero_like(usm_ndarray ary):
    """
    Make C-contiguous array of zero elements with same shape
    and type as ary.
    """
    cdef usm_ndarray r = usm_ndarray(
        _make_int_tuple(ary.nd_, ary.shape_),
        dtype=_make_typestr(ary.typenum_),
        buffer=ary.base_.get_usm_type()
    )
    # TODO: call function to set array elements to zero
    return r
