from numpy.core.numeric import normalize_axis_tuple

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti


# can be refactored later into general reduction
def _boolean_reduction(x, axis, keepdims, func):
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")

    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    axis = normalize_axis_tuple(axis, nd, "axis")

    exec_q = x.sycl_queue
    res_usm_type = x.usm_type

    red_nd = len(axis)
    if red_nd == 0:
        return dpt.astype(x, dpt.bool)

    perm = [i for i in range(nd) if i not in axis] + list(axis)
    x_tmp = dpt.permute_dims(x, perm)
    res_shape = x_tmp.shape[: nd - red_nd]

    wait_list = []
    res_tmp = dpt.empty(
        res_shape,
        dtype=dpt.int32,
        usm_type=res_usm_type,
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


def all(x, axis=None, out=None, keepdims=False):
    return _boolean_reduction(x, axis, keepdims, ti._all)


def any(x, axis=None, keepdims=False):
    return _boolean_reduction(x, axis, keepdims, ti._any)
