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

from cpython.mem cimport PyMem_Malloc
from cpython.ref cimport Py_INCREF
from cpython.tuple cimport PyTuple_New, PyTuple_SetItem


cdef int ERROR_MALLOC = 1
cdef int ERROR_INTERNAL = -1
cdef int ERROR_INCORRECT_ORDER = 2
cdef int ERROR_UNEXPECTED_STRIDES = 3

cdef int USM_ARRAY_C_CONTIGUOUS = 1
cdef int USM_ARRAY_F_CONTIGUOUS = 2
cdef int USM_ARRAY_WRITEABLE = 4


cdef Py_ssize_t shape_to_elem_count(int nd, Py_ssize_t *shape_arr):
    """
    Computes number of elements in an array.
    """
    cdef Py_ssize_t count = 1
    for i in range(nd):
        count *= shape_arr[i]
    return count


cdef int _from_input_shape_strides(
    int nd, object shape, object strides, int itemsize, char order,
    Py_ssize_t **shape_ptr, Py_ssize_t **strides_ptr,
    Py_ssize_t *nelems, Py_ssize_t *min_disp, Py_ssize_t *max_disp,
    int *contig):
    """
    Arguments: nd, shape, strides, itemsize, order
    Modifies:
        shape_ptr - pointer to C array for shape values
        stride_ptr - pointer to C array for strides values
        nelems - Number of elements in array
        min_disp = min( dot(strides, index), index for shape)
        max_disp = max( dor(strides, index), index for shape)
        contig = enumation for array contiguity
    Returns: 0 on success, error code otherwise.
        On success pointers point to allocated arrays,
        Otherwise they are set to NULL
    """
    cdef int i
    cdef int j
    cdef int all_incr = 1
    cdef int all_decr = 1
    cdef Py_ssize_t elem_count = 1
    cdef Py_ssize_t min_shift = 0
    cdef Py_ssize_t max_shift = 0
    cdef Py_ssize_t str_i
    cdef Py_ssize_t* shape_arr
    cdef Py_ssize_t* strides_arr

    # 0-d array
    if (nd == 0):
        contig[0] = (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)
        nelems[0] = 1
        min_disp[0] = 0
        max_disp[0] = 0
        shape_ptr[0] = <Py_ssize_t *>(<size_t>0)
        strides_ptr[0] = <Py_ssize_t *>(<size_t>0)
        return 0

    shape_arr = <Py_ssize_t*>PyMem_Malloc(nd * sizeof(Py_ssize_t))
    if (not shape_arr):
        return ERROR_MALLOC
    shape_ptr[0] = shape_arr
    for i in range(0, nd):
        shape_arr[i] = <Py_ssize_t> shape[i]
        elem_count *= shape_arr[i]
    if elem_count == 0:
        contig[0] = (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)
        nelems[0] = 1
        min_disp[0] = 0
        max_disp[0] = 0
        if strides is None:
            strides_ptr[0] = <Py_ssize_t *>(<size_t>0)
        else:
            strides_arr = <Py_ssize_t*>PyMem_Malloc(nd * sizeof(Py_ssize_t))
            if (not strides_arr):
                PyMem_Free(shape_ptr[0]);
                shape_ptr[0] = <Py_ssize_t *>(<size_t>0)
                return ERROR_MALLOC
            strides_ptr[0] = strides_arr
            for i in range(0, nd):
                strides_arr[i] = <Py_ssize_t> strides[i]
        return 0
    nelems[0] = elem_count
    if (strides is None):
        # no need to allocate and populate strides
        if (int(order) not in [ord('C'), ord('F'), ord('c'), ord('f')]):
            PyMem_Free(shape_ptr[0]);
            shape_ptr[0] = <Py_ssize_t *>(<size_t>0)
            return ERROR_INCORRECT_ORDER
        if order == <char> ord('C') or order == <char> ord('c'):
            contig[0] = USM_ARRAY_C_CONTIGUOUS
        else:
            contig[0] = USM_ARRAY_F_CONTIGUOUS
        if nd == 1:
            contig[0] = USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS
        else:
            j = 0
            for i in range(nd):
                if shape_arr[i] > 1:
                    j = j + 1
            if j < 2:
                contig[0] = USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS
        min_disp[0] = 0
        max_disp[0] = (elem_count - 1)
        strides_ptr[0] = <Py_ssize_t *>(<size_t>0)
        return 0
    elif ((isinstance(strides, (list, tuple)) or hasattr(strides, 'tolist'))
          and len(strides) == nd):
        strides_arr = <Py_ssize_t*>PyMem_Malloc(nd * sizeof(Py_ssize_t))
        if (not strides_arr):
            PyMem_Free(shape_ptr[0]);
            shape_ptr[0] = <Py_ssize_t *>(<size_t>0)
            return ERROR_MALLOC
        strides_ptr[0] = strides_arr
        for i in range(0, nd):
            str_i = <Py_ssize_t> strides[i]
            strides_arr[i] = str_i
            if str_i > 0:
                max_shift += strides_arr[i] * (shape_arr[i] - 1)
            else:
                min_shift += strides_arr[i] * (shape_arr[i] - 1)
        min_disp[0] = min_shift
        max_disp[0] = max_shift
        if max_shift == min_shift + (elem_count - 1):
            if elem_count == 1:
                contig[0] = (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)
                return 0
            if nd == 1:
                if strides_arr[0] == 1:
                    contig[0] = USM_ARRAY_C_CONTIGUOUS
                else:
                    contig[0] = 0
                return 0
            i = 0
            while i < nd:
                if shape_arr[i] == 1:
                    i = i + 1
                    continue
                j = i + 1
                while (j < nd and shape_arr[j] == 1):
                    j = j + 1
                if j < nd:
                    if all_incr:
                        all_incr = (
                            (strides_arr[i] > 0) and
                            (strides_arr[j] > 0) and
                            (strides_arr[i] <= strides_arr[j])
                        )
                    if all_decr:
                        all_decr = (
                            (strides_arr[i] > 0) and
                            (strides_arr[j] > 0) and
                            (strides_arr[i] >= strides_arr[j])
                        )
                    i = j
                else:
                    break
            if all_incr and all_decr:
                contig[0] = (USM_ARRAY_C_CONTIGUOUS | USM_ARRAY_F_CONTIGUOUS)
            elif all_incr:
                contig[0] = USM_ARRAY_F_CONTIGUOUS
            elif all_decr:
                contig[0] = USM_ARRAY_C_CONTIGUOUS
            else:
                contig[0] = 0
            return 0
        else:
            contig[0] = 0  # non-contiguous
        return 0
    else:
        PyMem_Free(shape_ptr[0]);
        shape_ptr[0] = <Py_ssize_t *>(<size_t>0)
        return ERROR_UNEXPECTED_STRIDES
    # return ERROR_INTERNAL


cdef object _make_int_tuple(int nd, Py_ssize_t *ary):
    """
    Makes Python tuple from C array
    """
    cdef tuple res
    cdef object tmp
    if (ary):
        res = PyTuple_New(nd)
        for i in range(nd):
            tmp = <object>ary[i]
            Py_INCREF(tmp)  # SetItem steals the reference
            PyTuple_SetItem(res, i, tmp)
        return res
    else:
        return None


cdef object _make_reversed_int_tuple(int nd, Py_ssize_t *ary):
    """
    Makes Python reversed tuple from C array
    """
    cdef tuple res
    cdef object tmp
    cdef int i
    cdef int nd_1
    if (ary):
        res = PyTuple_New(nd)
        nd_1 = nd - 1
        for i in range(nd):
            tmp = <object>ary[i]
            Py_INCREF(tmp)  # SetItem steals the reference
            PyTuple_SetItem(res, nd_1 - i, tmp)
        return res
    else:
        return None


cdef object _c_contig_strides(int nd, Py_ssize_t *shape):
    """
    Makes Python tuple for strides of C-contiguous array
    """
    cdef tuple cc_strides = PyTuple_New(nd)
    cdef object si = 1
    cdef int i
    cdef int nd_1 = nd - 1
    for i in range(0, nd):
        Py_INCREF(si)  # SetItem steals the reference
        PyTuple_SetItem(cc_strides, nd_1 - i, si)
        si = si * shape[nd_1 - i]
    return cc_strides


cdef object _f_contig_strides(int nd, Py_ssize_t *shape):
    """
    Makes Python tuple for strides of F-contiguous array
    """
    cdef tuple fc_strides = PyTuple_New(nd)
    cdef object si = 1
    for i in range(0, nd):
        Py_INCREF(si)  # SetItem steals the reference
        PyTuple_SetItem(fc_strides, i, si)
        si = si * shape[i]
    return fc_strides

cdef object _swap_last_two(tuple t):
    """
    Swap last two elements of a tuple
    """
    cdef int nd = len(t)
    cdef tuple res
    cdef int i
    cdef object tmp
    if (nd < 2):
        return t
    res = PyTuple_New(nd)
    # copy all elements except the last two
    for i in range(0, nd-2):
        tmp = t[i]
        Py_INCREF(tmp)  # SetItem steals the reference
        PyTuple_SetItem(res, i, tmp)
    # swap the last two elements
    tmp = t[nd-1]
    Py_INCREF(tmp)  # SetItem steals
    PyTuple_SetItem(res, nd - 2, tmp)
    tmp = t[nd-2]
    Py_INCREF(tmp)  # SetItem steals
    PyTuple_SetItem(res, nd - 1, tmp)
    return res
