# distutils: language = c++
# cython: language_level=3

import numpy as np

import dpctl
import dpctl.memory as dpmem

from cpython.mem cimport PyMem_Free
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem


cdef extern from "usm_array.hpp" namespace "usm_array":
    cdef cppclass usm_array:
        usm_array(char *, int, size_t*, Py_ssize_t *,
                  int, int, DPCTLSyclQueueRef) except +


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
        self.base = None
        self.nd = -1
        self.data = <char *>0
        self.shape = <Py_ssize_t *>0
        self.strides = <Py_ssize_t *>0
        self.flags = 0

    cdef void _cleanup(usm_ndarray self):
        if (self.shape):
            PyMem_Free(self.shape)
        if (self.strides):
            PyMem_Free(self.strides)
        self._reset()

    cdef usm_ndarray _clone(self):
        """
        Provides a copy of Python object pointing to the same data
        """
        cdef int item_size = type_bytesize(self.typenum)
        cdef Py_ssize_t offset_bytes = (
            (<char *> self.data) -
            (<char *>(<size_t>self.base._pointer)))
        cdef usm_ndarray res = usm_ndarray.__new__(
            usm_ndarray, _make_int_tuple(self.nd, self.shape),
            dtype=_make_typestr(self.typenum),
            strides=(
                _make_int_tuple(self.nd, self.strides) if (self.strides)
                else None),
            buffer=self.base,
            offset=(offset_bytes // item_size),
            order=('C' if (self.flags & USM_ARRAY_C_CONTIGUOUS) else 'F')
        )
        res.flags = self.flags
        if (res.data != self.data):
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
                    "or an object with __sycl_usm_array_interface__ "
                    "property".format(buffer))
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
        self.base = _buffer
        self.data = (<char *> (<size_t> _buffer._pointer)) + itemsize * _offset
        self.shape = shape_ptr
        self.strides = strides_ptr
        self.typenum = typenum
        self.flags = contig_flag
        self.nd = nd

    def __dealloc__(self):
        self._cleanup()

    cdef Py_ssize_t get_offset(self) except *:
        cdef char *mem_ptr = NULL
        cdef char *ary_ptr = self.data
        mem_ptr = <char *>(<size_t> self.base._pointer)
        byte_offset = ary_ptr - mem_ptr
        item_size = type_bytesize(self.typenum)
        if (byte_offset % item_size):
            raise InternalUSMArrayError(
                "byte_offset is not a multiple of item_size.")
        return byte_offset // item_size

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
        if (not isinstance(self.base, dpmem._memory._Memory)):
            raise ValueError("Invalid instance of usm_ndarray ecountered")
        ary_iface = self.base.__sycl_usm_array_interface__
        mem_ptr = <char *>(<size_t> ary_iface['data'][0])
        ary_ptr = <char *>(<size_t> self.data)
        ro_flag = False if (self.flags & USM_ARRAY_WRITEABLE) else True
        ary_iface['data'] = (<size_t> ary_ptr, ro_flag)
        ary_iface['shape'] = _make_int_tuple(self.nd, self.shape)
        if (self.strides):
            ary_iface['strides'] = _make_int_tuple(self.nd, self.strides)
        else:
            if (self.flags & USM_ARRAY_C_CONTIGUOUS):
                ary_iface['strides'] = None
            elif (self.flags & USM_ARRAY_F_CONTIGUOUS):
                ary_iface['strides'] = _f_contig_strides(self.nd, self.shape)
            else:
                raise ValueError("USM Array is not contiguous and "
                                 "has empty strides")
        ary_iface['typestr'] = _make_typestr(self.typenum)
        byte_offset = ary_ptr - mem_ptr
        item_size = type_bytesize(self.typenum)
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
        return int(self.nd)

    @property
    def usm_data(self):
        """
        Gives USM memory object underlying usm_array instance.
        """
        return self.base

    @property
    def shape(self):
        """
        Elements of the shape tuple give the lengths of the
        respective array dimensions.
        """
        return _make_int_tuple(self.nd, self.shape) if self.nd > 0 else tuple()

    @property
    def strides(self):
        """
        Returns memory displacement in array elements, upon unit
        change of respective index.

        E.g. for strides (s1, s2, s3) and multi-index (i1, i2, i3)

           a[i1, i2, i3] == (&a[0,0,0])[ s1*s1 + s2*i2 + s3*i3]
        """
        if (self.strides):
            return _make_int_tuple(self.nd, self.strides)
        else:
            if (self.flags & USM_ARRAY_C_CONTIGUOUS):
                return _c_contig_strides(self.nd, self.shape)
            elif (self.flags & USM_ARRAY_F_CONTIGUOUS):
                return _f_contig_strides(self.nd, self.shape)
            else:
                raise ValueError("Inconsitent usm_ndarray data")

    @property
    def flags(self):
        """
        Currently returns integer whose bits correspond to the flags.
        """
        return int(self.flags)

    @property
    def usm_type(self):
        """
        USM type of underlying memory. Can be 'device', 'shared', or 'host'.

        See: https://docs.oneapi.com/versions/latest/dpcpp/iface/usm.html
        """
        return self.base.get_usm_type()

    @property
    def itemsize(self):
        """
        Size of array element in bytes.
        """
        return type_bytesize(self.typenum)

    @property
    def nbytes(self):
        """
        Total bytes consumed by the elements of the array.
        """
        return (
            shape_to_elem_count(self.nd, self.shape) *
            type_bytesize(self.typenum))

    @property
    def size(self):
        """
        Number of elements in the array.
        """
        return shape_to_elem_count(self.nd, self.shape)

    @property
    def dtype(self):
        """
        Returns NumPy's dtype corresponding to the type of the array elements.
        """
        return np.dtype(_make_typestr(self.typenum))

    @property
    def sycl_queue(self):
        """
        Returns `dpctl.SyclQueue` object associated with USM data.
        """
        return self.base._queue

    @property
    def sycl_device(self):
        """
        Returns `dpctl.SyclDevice` object on which USM data was allocated.
        """
        return self.base._queue.sycl_device

    @property
    def sycl_context(self):
        """
        Returns `dpctl.SyclContext` object to which USM data is bound.
        """
        return self.base._queue.sycl_context

    @property
    def T(self):
        if self.nd < 2:
            return self
        else:
            return _transpose(self)

    @property
    def real(self):
        if (self.typenum < UAR_CFLOAT):
            # elements are real
            return self
        if (self.typenum < UAR_TYPE_SENTINEL):
            return _real_view(self)

    @property
    def imag(self):
        if (self.typenum < UAR_CFLOAT):
            # elements are real
            return _zero_like(self)
        if (self.typenum < UAR_TYPE_SENTINEL):
            return _imag_view(self)

    def __getitem__(self, ind):
        cdef tuple _meta = _basic_slice_meta(
            ind, (<object>self).shape, (<object> self).strides,
            self.get_offset())
        cdef usm_ndarray res

        res = usm_ndarray.__new__(
            usm_ndarray, _meta[0],
            dtype=_make_typestr(self.typenum),
            strides=_meta[1],
            buffer=self.base,
            offset=_meta[2]
        )
        res.flags |= (self.flags & USM_ARRAY_WRITEABLE)
        return res


cdef usm_ndarray _real_view(usm_ndarray ary):
    """
    View into real parts of a complex type array
    """
    cdef usm_ndarray r = ary._clone()
    if (ary.typenum == UAR_CFLOAT):
        r.typenum = UAR_FLOAT
    elif (ary.typenum == UAR_CDOUBLE):
        r.typenum = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_real_view call on array of non-complex type.")
    return r


cdef usm_ndarray _imag_view(usm_ndarray ary):
    """
    View into imaginary parts of a complex type array
    """
    cdef usm_ndarray r = ary._clone()
    if (ary.typenum == UAR_CFLOAT):
        r.typenum = UAR_FLOAT
    elif (ary.typenum == UAR_CDOUBLE):
        r.typenum = UAR_DOUBLE
    else:
        raise InternalUSMArrayError(
            "_real_view call on array of non-complex type.")
    # displace pointer to imaginary part
    r.data = r.data + type_bytesize(r.typenum)
    return r


cdef usm_ndarray _transpose(usm_ndarray ary):
    """
    Construct transposed array without copying the data
    """
    cdef usm_ndarray r = usm_ndarray.__new__(
        usm_ndarray,
        _make_reversed_int_tuple(ary.nd, ary.shape),
        dtype=_make_typestr(ary.typenum),
        strides=(
            _make_reversed_int_tuple(ary.nd, ary.strides)
            if (ary.strides) else None),
        buffer=ary.base,
        order=('F' if (ary.flags & USM_ARRAY_C_CONTIGUOUS) else 'C')
    )
    r.flags |= (ary.flags & USM_ARRAY_WRITEABLE)
    return r


cdef usm_ndarray _zero_like(usm_ndarray ary):
    """
    Make C-contiguous array of zero elements with same shape
    and type as ary.
    """
    cdef usm_ndarray r = usm_ndarray(
        _make_int_tuple(ary.nd, ary.shape),
        dtype=_make_typestr(ary.typenum),
        buffer=ary.base.get_usm_type()
    )
    # TODO: call function to set array elements to zero
    return r
