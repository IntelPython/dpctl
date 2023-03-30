#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
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

cdef bint _valid_usm_ptr_and_context(DPCTLSyclUSMRef ptr, SyclContext ctx):
    usm_type = _Memory.get_pointer_type(ptr, ctx)
    return usm_type in (b"shared", b"device", b"host")


cdef DPCTLSyclQueueRef _queue_ref_copy_from_SyclQueue(
    DPCTLSyclUSMRef ptr, SyclQueue q):
    """ Check that USM ptr is consistent with SYCL context in the queue,
    and return a copy of QueueRef if so, or NULL otherwise.
    """
    cdef SyclContext ctx = q.get_sycl_context()
    if (_valid_usm_ptr_and_context(ptr, ctx)):
        return DPCTLQueue_Copy(q.get_queue_ref())
    else:
        return NULL


cdef DPCTLSyclQueueRef _queue_ref_copy_from_USMRef_and_SyclContext(
    DPCTLSyclUSMRef ptr, SyclContext ctx):
    """ Obtain device from pointer and sycl context, use
        context and device to create a queue from which this memory
        can be accessible.
    """
    cdef SyclDevice dev = _Memory.get_pointer_device(ptr, ctx)
    cdef DPCTLSyclContextRef CRef = ctx.get_context_ref()
    cdef DPCTLSyclDeviceRef DRef = dev.get_device_ref()
    return DPCTLQueue_Create(CRef, DRef, NULL, 0)


cdef DPCTLSyclQueueRef get_queue_ref_from_ptr_and_syclobj(
    DPCTLSyclUSMRef ptr, object syclobj):
    """ Constructs queue from pointer and syclobject from
        __sycl_usm_array_interface__
    """
    cdef SyclContext ctx
    if type(syclobj) is SyclQueue:
        return _queue_ref_copy_from_SyclQueue(ptr, <SyclQueue> syclobj)
    elif type(syclobj) is SyclContext:
        ctx = <SyclContext>syclobj
        return _queue_ref_copy_from_USMRef_and_SyclContext(ptr, ctx)
    elif type(syclobj) is str:
        q = SyclQueue(syclobj)
        return _queue_ref_copy_from_SyclQueue(ptr, <SyclQueue> q)
    elif pycapsule.PyCapsule_IsValid(syclobj, "SyclQueueRef"):
        q = SyclQueue(syclobj)
        return _queue_ref_copy_from_SyclQueue(ptr, <SyclQueue> q)
    elif pycapsule.PyCapsule_IsValid(syclobj, "SyclContextRef"):
        ctx = <SyclContext>SyclContext(syclobj)
        return _queue_ref_copy_from_USMRef_and_SyclContext(ptr, ctx)
    elif hasattr(syclobj, "_get_capsule"):
        cap = syclobj._get_capsule()
        if pycapsule.PyCapsule_IsValid(cap, "SyclQueueRef"):
            q = SyclQueue(cap)
            return _queue_ref_copy_from_SyclQueue(ptr, <SyclQueue> q)
        elif pycapsule.PyCapsule_IsValid(cap, "SyclContextRef"):
            ctx = <SyclContext>SyclContext(cap)
            return _queue_ref_copy_from_USMRef_and_SyclContext(ptr, ctx)
        else:
            return NULL
    else:
        return NULL


cdef object _pointers_from_shape_and_stride(
    int nd, object ary_shape, Py_ssize_t itemsize, Py_ssize_t ary_offset,
    object ary_strides):
    """
    Internal utility: for given array data about shape/layout/element
    compute left-most displacement when enumerating all elements of the array
    and the number of bytes of memory between the left-most and right-most
    displacements.

    Returns: tuple(min_disp, nbytes)
    """
    cdef Py_ssize_t nelems = 1
    cdef Py_ssize_t min_disp = 0
    cdef Py_ssize_t max_disp = 0
    cdef int i
    cdef Py_ssize_t sh_i = 0
    cdef Py_ssize_t str_i = 0
    if (nd > 0):
        if (ary_strides is None):
            nelems = 1
            for si in ary_shape:
                sh_i = int(si)
                if (sh_i < 0):
                    raise ValueError("Array shape elements need to be positive")
                nelems = nelems * sh_i
            return (ary_offset, max(nelems, 1) * itemsize)
        else:
            min_disp = ary_offset
            max_disp = ary_offset
            for i in range(nd):
                str_i = int(ary_strides[i])
                sh_i = int(ary_shape[i])
                if (sh_i < 0):
                    raise ValueError("Array shape elements need to be positive")
                if (sh_i > 0):
                    if (str_i > 0):
                        max_disp += str_i * (sh_i - 1)
                    else:
                        min_disp += str_i * (sh_i - 1)
                else:
                    nelems = 0
            if nelems == 0:
                return (ary_offset, itemsize)
            return (min_disp, (max_disp - min_disp + 1) * itemsize)
    elif (nd == 0):
        return (ary_offset, itemsize)
    else:
        raise ValueError("Array dimensions can not be negative")


cdef class _USMBufferData:
    """
    Internal data struct populated from parsing
    `__sycl_usm_array_interface__` dictionary
    """
    cdef DPCTLSyclUSMRef p
    cdef int writable
    cdef object dt
    cdef Py_ssize_t itemsize
    cdef Py_ssize_t nbytes
    cdef SyclQueue queue

    @staticmethod
    cdef _USMBufferData from_sycl_usm_ary_iface(dict ary_iface):
        cdef object ary_data_tuple = ary_iface.get('data', None)
        cdef object ary_typestr = ary_iface.get('typestr', None)
        cdef object ary_shape = ary_iface.get('shape', None)
        cdef object ary_strides = ary_iface.get('strides', None)
        cdef object ary_syclobj = ary_iface.get('syclobj', None)
        cdef Py_ssize_t ary_offset = ary_iface.get('offset', 0)
        cdef int ary_version = ary_iface.get('version', 0)
        cdef size_t arr_data_ptr = 0
        cdef DPCTLSyclUSMRef memRef = NULL
        cdef Py_ssize_t itemsize = -1
        cdef int writable = -1
        cdef int nd = -1
        cdef DPCTLSyclQueueRef QRef = NULL
        cdef object dt
        cdef _USMBufferData buf
        cdef SyclDevice dev
        cdef SyclContext ctx

        if ary_version != 1:
            raise ValueError(("__sycl_usm_array_interface__ is malformed:"
                              " dict('version': {}) is unexpected."
                              " The only recognized version is 1.").format(
                                  ary_version))
        if not ary_data_tuple or len(ary_data_tuple) != 2:
            raise ValueError("__sycl_usm_array_interface__ is malformed:"
                             " 'data' field is required, and must be a tuple"
                             " (usm_pointer, is_writable_boolean).")
        arr_data_ptr = <size_t>ary_data_tuple[0]
        writable = 1 if ary_data_tuple[1] else 0
        # Check that memory and syclobj are consistent:
        # (USM pointer is bound to this sycl context)
        memRef = <DPCTLSyclUSMRef>arr_data_ptr
        QRef = get_queue_ref_from_ptr_and_syclobj(memRef, ary_syclobj)
        if (QRef is NULL):
            raise ValueError("__sycl_usm_array_interface__ is malformed:"
                             " 'data' field is not consistent with 'syclobj'"
                             " field, the pointer {} is not bound to"
                             " SyclContext derived from"
                             " dict('syclobj': {}).".format(
                                 hex(arr_data_ptr), ary_syclobj))
        # shape must be present
        if ary_shape is None or not (
                isinstance(ary_shape, collections.abc.Sized) and
                isinstance(ary_shape, collections.abc.Iterable)):
            DPCTLQueue_Delete(QRef)
            raise ValueError("Shape entry is a required element of "
                             "`__sycl_usm_array_interface__` dictionary")
        nd = len(ary_shape)
        try:
            dt = np.dtype(ary_typestr)
            if (dt.hasobject or not (np.issubdtype(dt.type, np.number) or
                                     dt.type is np.bool_)):
                DPCTLQueue_Delete(QRef)
                raise TypeError("Only integer types, floating and complex "
                                "floating types are supported.")
            itemsize = <Py_ssize_t> (dt.itemsize)
        except TypeError as e:
            raise ValueError(
                "__sycl_usm_array_interface__ is malformed:"
                " dict('typestr': {}) is unexpected. ".format(ary_typestr)
            ) from e

        if (ary_strides is None or (
                isinstance(ary_strides, collections.abc.Sized) and
                isinstance(ary_strides, collections.abc.Iterable) and
                len(ary_strides) == nd)):
            min_disp, nbytes = _pointers_from_shape_and_stride(
                nd, ary_shape, itemsize, ary_offset, ary_strides)
        else:
            DPCTLQueue_Delete(QRef)
            raise ValueError("__sycl_usm_array_interface__ is malformed: "
                             "'strides' must be a tuple or "
                             "list of the same length as shape")

        buf = _USMBufferData.__new__(_USMBufferData)
        buf.p = <DPCTLSyclUSMRef>(
            arr_data_ptr + (<Py_ssize_t>min_disp) * itemsize)
        buf.writable = writable
        buf.itemsize = itemsize
        buf.nbytes = <Py_ssize_t> nbytes

        buf.queue = SyclQueue._create(QRef)

        return buf
