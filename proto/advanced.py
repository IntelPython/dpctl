import numbers

import dpctl.tensor as dpt
import dpctl.utils
from dpctl.tensor import usm_ndarray

"""
Advanced slicing meta-infomation extraction
"""


class ExecutionPlacementError(Exception):
    pass


def _slice_len(sl_start: int, sl_stop: int, sl_step: int):
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


def _is_integral(x):
    """Gives True if x is an integral slice spec"""
    if isinstance(x, (int, numbers.Integral)):
        return True
    if isinstance(x, usm_ndarray):
        if x.ndim > 0:
            return False
        if x.dtype.kind not in "ui":
            return False
        return True
    if callable(getattr(x, "__index__", None)):
        try:
            x.__index__()
        except (TypeError, ValueError):
            return False
        return True
    return False


def _basic_slice_meta(ind, shape: tuple, strides: tuple, offset: int):
    """
    Give basic slicing index `ind` and array layout information produce
    a 5-tuple (resulting_shape, resulting_strides, resulting_offset,
       advanced_ind, resulting_advanced_ind_pos)
    used to contruct a view into underlying array over which advanced
    indexing, if any, is to be performed.

    Raises IndexError for invalid index `ind`.
    """
    _no_advanced_ind = tuple()
    _no_advanced_pos = -1
    if ind is Ellipsis:
        return (shape, strides, offset, _no_advanced_ind, _no_advanced_pos)
    elif ind is None:
        return (
            (1,) + shape,
            (0,) + strides,
            offset,
            _no_advanced_ind,
            _no_advanced_pos,
        )
    elif isinstance(ind, slice):
        sl_start, sl_stop, sl_step = ind.indices(shape[0])
        sh0 = _slice_len(sl_start, sl_stop, sl_step)
        str0 = sl_step * strides[0]
        new_strides = (
            strides if (sl_step == 1 or sh0 == 0) else (str0,) + strides[1:]
        )
        new_offset = offset if sh0 == 0 else offset + sl_start * strides[0]
        return (
            (sh0,) + shape[1:],
            new_strides,
            new_offset,
            _no_advanced_ind,
            _no_advanced_pos,
        )
    elif _is_integral(ind):
        ind = ind.__index__()
        if 0 <= ind < shape[0]:
            return (
                shape[1:],
                strides[1:],
                offset + ind * strides[0],
                _no_advanced_ind,
                _no_advanced_pos,
            )
        elif -shape[0] <= ind < 0:
            return (
                shape[1:],
                strides[1:],
                offset + (shape[0] + ind) * strides[0],
                _no_advanced_ind,
                _no_advanced_pos,
            )
        else:
            raise IndexError(
                "Index {0} is out of range for axes 0 with "
                "size {1}".format(ind, shape[0])
            )
    elif isinstance(ind, usm_ndarray):
        return (shape, strides, 0, (ind,), 0)
    elif isinstance(ind, tuple):
        axes_referenced = 0
        ellipses_count = 0
        newaxis_count = 0
        explicit_index = 0
        array_count = 0
        seen_arrays_yet = False
        array_streak_started = False
        array_streak_interrupted = False
        for i in ind:
            if i is None:
                newaxis_count += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif i is Ellipsis:
                ellipses_count += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif isinstance(i, slice):
                axes_referenced += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif _is_integral(i):
                explicit_index += 1
                axes_referenced += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif isinstance(i, usm_ndarray):
                if not seen_arrays_yet:
                    seen_arrays_yet = True
                    array_streak_started = True
                    array_streak_interrupted = False
                if array_streak_interrupted:
                    raise IndexError(
                        "Advanced indexing array specs may not be "
                        "separated by basic slicing specs."
                    )
                dt_k = i.dtype.kind
                if dt_k == "b":
                    axes_referenced += i.ndim
                elif dt_k in "ui":
                    axes_referenced += 1
                else:
                    raise IndexError(
                        "arrays used as indices must be of integer "
                        "(or boolean) type"
                    )
                array_count += 1
            else:
                raise TypeError
        if ellipses_count > 1:
            raise IndexError("an index can only have a sinlge ellipsis ('...')")
        if axes_referenced > len(shape):
            raise IndexError(
                "too many indices for an array, array is "
                "{0}-dimensional, but {1} were indexed".format(
                    len(shape), axes_referenced
                )
            )
        if ellipses_count:
            ellipses_count = len(shape) - axes_referenced
        new_shape_len = (
            newaxis_count + ellipses_count + axes_referenced - explicit_index
        )
        new_shape = list()
        new_strides = list()
        new_advanced_ind = list()
        k = 0
        new_advanced_start_pos = -1
        advanced_start_pos_set = False
        new_offset = offset
        is_empty = False
        for i in range(len(ind)):
            ind_i = ind[i]
            if ind_i is Ellipsis:
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
            elif _is_integral(ind_i):
                ind_i = ind_i.__index__()
                if 0 <= ind_i < shape[k]:
                    k_new = k + 1
                    if not is_empty:
                        new_offset = new_offset + ind_i * strides[k]
                    k = k_new
                elif -shape[k] <= ind_i < 0:
                    k_new = k + 1
                    if not is_empty:
                        new_offset = (
                            new_offset + (shape[k] + ind_i) * strides[k]
                        )
                    k = k_new
                else:
                    raise IndexError(
                        (
                            "Index {0} is out of range for "
                            "axes {1} with size {2}"
                        ).format(ind_i, k, shape[k])
                    )
            elif isinstance(ind_i, usm_ndarray):
                if not advanced_start_pos_set:
                    new_advanced_start_pos = len(new_shape)
                    advanced_start_pos_set = True
                new_advanced_ind.append(ind_i)
                dt_k = ind_i.dtype.kind
                if dt_k == "b":
                    k_new = k + ind_i.ndim
                else:
                    k_new = k + 1
                new_shape.extend(shape[k:k_new])
                new_strides.extend(strides[k:k_new])
                k = k_new
        new_shape.extend(shape[k:])
        new_strides.extend(strides[k:])
        debug = True
        if debug:
            new_shape_len += len(shape) - k
            assert (
                len(new_shape) == new_shape_len
            ), f"{len(new_shape)} vs {new_shape_len}"
            assert (
                len(new_strides) == new_shape_len
            ), f"{len(new_strides)} vs {new_shape_len}"
            assert len(new_advanced_ind) == array_count
        return (
            tuple(new_shape),
            tuple(new_strides),
            new_offset,
            tuple(new_advanced_ind),
            new_advanced_start_pos,
        )
    else:
        raise TypeError


def _mock_extract(ary, ary_mask, p):
    exec_q = dpctl.utils.get_execution_queue(
        (
            ary.sycl_queue,
            ary_mask.sycl_queue,
        )
    )
    if exec_q is None:
        raise ExecutionPlacementError(
            "Can not automatically determine where to allocate the "
            "result or performance execution. "
            "Use `usm_ndarray.to_device` method to migrate data to "
            "be associated with the same queue."
        )

    res_usm_type = dpctl.utils.get_coerced_usm_type(
        (
            ary.usm_type,
            ary_mask.usm_type,
        )
    )
    ary_np = dpt.asnumpy(ary)
    mask_np = dpt.asnumpy(ary_mask)
    res_np = ary_np[(slice(None),) * p + (mask_np,)]
    res = dpt.empty(
        res_np.shape, dtype=ary.dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )
    res[...] = res_np
    return res


def _mock_nonzero(ary):
    if not isinstance(ary, usm_ndarray):
        raise TypeError
    q = ary.sycl_queue
    usm_type = ary.usm_type
    ary_np = dpt.asnumpy(ary)
    nz = ary_np.nonzero()
    return tuple(dpt.asarray(i, usm_type=usm_type, sycl_queue=q) for i in nz)


def _mock_take_multi_index(ary, inds, p):
    queues_ = [
        ary.sycl_queue,
    ]
    usm_types_ = [
        ary.usm_type,
    ]
    all_integers = True
    for ind in inds:
        queues_.append(ind.sycl_queue)
        usm_types_.append(ind.usm_type)
        if all_integers:
            all_integers = ind.dtype.kind in "ui"
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise ExecutionPlacementError("")
    if not all_integers:
        print(inds)
        raise IndexError(
            "arrays used as indices must be of integer (or boolean) type"
        )
    ary_np = dpt.asnumpy(ary)
    ind_np = (slice(None),) * p + tuple(dpt.asnumpy(ind) for ind in inds)
    res_np = ary_np[ind_np]
    res_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)
    res = dpt.empty(
        res_np.shape, dtype=ary.dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )
    res[...] = res_np
    return res


def get_item(ary, ind):
    suai = ary.__sycl_usm_array_interface__
    _meta = _basic_slice_meta(
        ind, ary.shape, ary.strides, suai.get("offset", 0)
    )

    if len(_meta) < 5:
        raise RuntimeError

    res = usm_ndarray.__new__(
        usm_ndarray,
        _meta[0],
        dtype=ary.dtype,  # _make_typestr(ary.dtype.num),
        strides=_meta[1],
        buffer=ary.usm_data,  # self.base_,
        offset=_meta[2],
    )
    # set flags and namespace
    # res.flags_ |= (ary.flags_ & USM_ARRAY_WRITABLE)
    # res.array_namespace_ = self.array_namespace_
    adv_ind = _meta[3]
    adv_ind_start_p = _meta[4]

    if adv_ind_start_p < 0:
        return res

    if len(adv_ind) == 1 and adv_ind[0].dtype == dpt.bool:
        return _mock_extract(res, adv_ind[0], adv_ind_start_p)

    if any(ind.dtype == dpt.bool for ind in adv_ind):
        adv_ind_int = list()
        for ind in adv_ind:
            if ind.dtype == dpt.bool:
                adv_ind_int.extend(_mock_nonzero(ind))
            else:
                adv_ind_int.append(ind)
        return _mock_take_multi_index(res, tuple(adv_ind_int), adv_ind_start_p)

    return _mock_take_multi_index(res, adv_ind, adv_ind_start_p)


def _mock_place(ary, ary_mask, p, vals):
    exec_q = dpctl.utils.get_execution_queue(
        (ary.sycl_queue, ary_mask.sycl_queue, vals.sycl_queue)
    )
    if exec_q is None:
        raise ExecutionPlacementError(
            "Can not automatically determine where to allocate the "
            "result or performance execution. "
            "Use `usm_ndarray.to_device` method to migrate data to "
            "be associated with the same queue."
        )

    ary_np = dpt.asnumpy(ary)
    mask_np = dpt.asnumpy(ary_mask)
    vals_np = dpt.asnumpy(vals)
    ary_np[(slice(None),) * p + (mask_np,)] = vals_np
    ary[...] = ary_np
    return


def _mock_put_multi_index(ary, inds, p, vals):
    queues_ = [ary.sycl_queue, vals.sycl_queue]
    usm_types_ = [ary.usm_type, vals.usm_type]
    all_integers = True
    for ind in inds:
        queues_.append(ind.sycl_queue)
        usm_types_.append(ind.usm_type)
        if all_integers:
            all_integers = ind.dtype.kind in "ui"
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise ExecutionPlacementError("")
    if not all_integers:
        print(inds)
        raise IndexError(
            "arrays used as indices must be of integer (or boolean) type"
        )
    ary_np = dpt.asnumpy(ary)
    vals_np = dpt.asnumpy(vals)
    ind_np = (slice(None),) * p + tuple(dpt.asnumpy(ind) for ind in inds)
    ary_np[ind_np] = vals_np
    ary[...] = ary_np
    return


def set_item(ary, ind, rhs):
    suai = ary.__sycl_usm_array_interface__
    _meta = _basic_slice_meta(
        ind, ary.shape, ary.strides, suai.get("offset", 0)
    )

    if len(_meta) < 5:
        raise RuntimeError

    res = usm_ndarray.__new__(
        usm_ndarray,
        _meta[0],
        dtype=ary.dtype,  # _make_typestr(ary.dtype.num),
        strides=_meta[1],
        buffer=ary.usm_data,  # self.base_,
        offset=_meta[2],
    )
    # set flags and namespace
    # res.flags_ |= (ary.flags_ & USM_ARRAY_WRITABLE)
    # res.array_namespace_ = self.array_namespace_
    adv_ind = _meta[3]
    adv_ind_start_p = _meta[4]

    if adv_ind_start_p < 0:
        res[...] = rhs
        return

    if len(adv_ind) == 1 and adv_ind[0].dtype == dpt.bool:
        _mock_place(res, adv_ind[0], adv_ind_start_p, rhs)
        return

    if any(ind.dtype == dpt.bool for ind in adv_ind):
        adv_ind_int = list()
        for ind in adv_ind:
            if ind.dtype == dpt.bool:
                adv_ind_int.extend(_mock_nonzero(ind))
            else:
                adv_ind_int.append(ind)
        _mock_put_multi_index(res, tuple(adv_ind_int), adv_ind_start_p, rhs)
        return

    _mock_put_multi_index(res, adv_ind, adv_ind_start_p, rhs)
    return
