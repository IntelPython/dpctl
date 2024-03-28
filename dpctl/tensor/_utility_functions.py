from numpy.core.numeric import normalize_axis_tuple

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import dpctl.tensor._tensor_reductions_impl as tri


def _boolean_reduction(x, axis, keepdims, func):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    nd = x.ndim
    if axis is None:
        red_nd = nd
        # case of a scalar
        if red_nd == 0:
            return dpt.astype(x, dpt.bool)
        x_tmp = x
        res_shape = tuple()
        perm = list(range(nd))
    else:
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        axis = normalize_axis_tuple(axis, nd, "axis")

        red_nd = len(axis)
        # check for axis=()
        if red_nd == 0:
            return dpt.astype(x, dpt.bool)
        perm = [i for i in range(nd) if i not in axis] + list(axis)
        x_tmp = dpt.permute_dims(x, perm)
        res_shape = x_tmp.shape[: nd - red_nd]

    exec_q = x.sycl_queue
    res_usm_type = x.usm_type

    wait_list = []
    # always allocate the temporary as
    # int32 and usm-device  to ensure that atomic updates
    # are supported
    res_tmp = dpt.empty(
        res_shape,
        dtype=dpt.int32,
        usm_type="device",
        sycl_queue=exec_q,
    )
    hev0, ev0 = func(
        src=x_tmp,
        trailing_dims_to_reduce=red_nd,
        dst=res_tmp,
        sycl_queue=exec_q,
    )
    wait_list.append(hev0)

    # copy to boolean result array
    res = dpt.empty(
        res_shape,
        dtype=dpt.bool,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
    )
    hev1, _ = ti._copy_usm_ndarray_into_usm_ndarray(
        src=res_tmp, dst=res, sycl_queue=exec_q, depends=[ev0]
    )
    wait_list.append(hev1)

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(dpt.reshape(res, res_shape), inv_perm)
    dpctl.SyclEvent.wait_for(wait_list)
    return res


def all(x, /, *, axis=None, keepdims=False):
    """all(x, axis=None, keepdims=False)

    Tests whether all input array elements evaluate to True along a given axis.

    Args:
        x (usm_ndarray): Input array.
        axis (Optional[Union[int, Tuple[int,...]]]): Axis (or axes)
            along which to perform a logical AND reduction.
            When `axis` is `None`, a logical AND reduction
            is performed over all dimensions of `x`.
            If `axis` is negative, the axis is counted from
            the last dimension to the first.
            Default: `None`.
        keepdims (bool, optional): If `True`, the reduced axes are included
            in the result as singleton dimensions, and the result is
            broadcastable to the input array shape.
            If `False`, the reduced axes are not included in the result.
            Default: `False`.

    Returns:
        usm_ndarray:
            An array with a data type of `bool`
            containing the results of the logical AND reduction.
    """
    return _boolean_reduction(x, axis, keepdims, tri._all)


def any(x, /, *, axis=None, keepdims=False):
    """any(x, axis=None, keepdims=False)

    Tests whether any input array elements evaluate to True along a given axis.

    Args:
        x (usm_ndarray): Input array.
        axis (Optional[Union[int, Tuple[int,...]]]): Axis (or axes)
            along which to perform a logical OR reduction.
            When `axis` is `None`, a logical OR reduction
            is performed over all dimensions of `x`.
            If `axis` is negative, the axis is counted from
            the last dimension to the first.
            Default: `None`.
        keepdims (bool, optional): If `True`, the reduced axes are included
            in the result as singleton dimensions, and the result is
            broadcastable to the input array shape.
            If `False`, the reduced axes are not included in the result.
            Default: `False`.

    Returns:
        usm_ndarray:
            An array with a data type of `bool`
            containing the results of the logical OR reduction.
    """
    return _boolean_reduction(x, axis, keepdims, tri._any)
