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

import numbers


cdef Py_ssize_t _slice_len(
    Py_ssize_t sl_start,
    Py_ssize_t sl_stop,
    Py_ssize_t sl_step
):
    """
    Compute len(range(sl_start, sl_stop, sl_step))
    """
    if sl_start == sl_stop:
        return 0
    if sl_step > 0:
        # 1 + argmax k such htat sl_start + sl_step*k < sl_stop
        return 1 + ((sl_stop - sl_start - 1) // sl_step)
    else:
        return 1 + ((sl_stop - sl_start + 1) // sl_step)


cdef object _basic_slice_meta(object ind, tuple shape,
                              tuple strides, Py_ssize_t offset):
    """
    Give basic slicing index `ind` and array layout information produce
    a tuple (resulting_shape, resulting_strides, resultin_offset)
    used to contruct a view into underlying array.

    Raises IndexError for invalid index `ind`, and NotImplementedError
    if `ind` is an array.
    """
    is_integral = lambda x: (
        isinstance(x, numbers.Integral) or callable(getattr(x, "__index__", None))
    )
    if ind is Ellipsis:
        return (shape, strides, offset)
    elif ind is None:
        return ((1,) + shape, (0,) + strides, offset)
    elif isinstance(ind, slice):
        sl_start, sl_stop, sl_step = ind.indices(shape[0])
        sh0 = _slice_len(sl_start, sl_stop, sl_step)
        str0 = sl_step * strides[0]
        new_strides = strides if (sl_step == 1 or sh0 == 0) else (str0,) + strides[1:]
        new_offset = offset if sh0 == 0 else offset + sl_start * strides[0]
        return (
            (sh0, ) + shape[1:],
            new_strides,
            new_offset
        )
    elif is_integral(ind):
        ind = ind.__index__()
        if 0 <= ind < shape[0]:
            return (shape[1:], strides[1:], offset + ind * strides[0])
        elif -shape[0] <= ind < 0:
            return (shape[1:], strides[1:],
                    offset + (shape[0] + ind) * strides[0])
        else:
            raise IndexError(
                "Index {0} is out of range for axes 0 with "
                "size {1}".format(ind, shape[0]))
    elif isinstance(ind, list):
        raise NotImplemented
    elif isinstance(ind, tuple):
        axes_referenced = 0
        ellipses_count = 0
        newaxis_count = 0
        explicit_index = 0
        for i in ind:
            if i is None:
                newaxis_count = newaxis_count + 1
            elif i is Ellipsis:
                ellipses_count = ellipses_count + 1
            elif isinstance(i, slice):
                axes_referenced = axes_referenced + 1
            elif is_integral(i):
                explicit_index = explicit_index + 1
                axes_referenced = axes_referenced + 1
            elif isinstance(i, list):
                raise NotImplemented
            else:
                raise TypeError
        if ellipses_count > 1:
            raise IndexError(
                "an index can only have a sinlge ellipsis ('...')")
        if axes_referenced > len(shape):
            raise IndexError(
                "too many indices for an array, array is "
                "{0}-dimensional, but {1} were indexed".format(
                    len(shape), axes_referenced))
        if ellipses_count:
            ellipses_count = len(shape) - axes_referenced
        new_shape_len = (newaxis_count + ellipses_count
                         + axes_referenced - explicit_index)
        new_shape = list()
        new_strides = list()
        k = 0
        new_offset = offset
        is_empty = False
        for i in range(len(ind)):
            ind_i = ind[i]
            if (ind_i is Ellipsis):
                k_new = k + ellipses_count
                new_shape.extend(shape[k:k_new])
                new_strides.extend(strides[k:k_new])
                k = k_new
            elif ind_i is None:
                new_shape.append(1)
                new_strides.append(0)
            elif isinstance(ind_i, slice):
                k_new = k + 1
                sl_start, sl_stop, sl_step = ind_i.indices(shape[k])
                sh_i = _slice_len(sl_start, sl_stop, sl_step)
                str_i = (1 if sh_i == 0 else sl_step) * strides[k]
                new_shape.append(sh_i)
                new_strides.append(str_i)
                if sh_i > 0 and not is_empty:
                    new_offset = new_offset + sl_start * strides[k]
                if sh_i == 0:
                    is_empty = True
                k = k_new
            elif is_integral(ind_i):
                ind_i = ind_i.__index__()
                if 0 <= ind_i < shape[k]:
                    k_new = k + 1
                    if not is_empty:
                        new_offset = new_offset + ind_i * strides[k]
                    k = k_new
                elif -shape[k] <= ind_i < 0:
                    k_new = k + 1
                    if not is_empty:
                        new_offset = new_offset + (shape[k] + ind_i) * strides[k]
                    k = k_new
                else:
                    raise IndexError(
                        ("Index {0} is out of range for "
                        "axes {1} with size {2}").format(ind_i, k, shape[k]))
        new_shape.extend(shape[k:])
        new_strides.extend(strides[k:])
        return (tuple(new_shape), tuple(new_strides), new_offset)
    else:
        raise TypeError
