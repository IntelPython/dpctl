import numpy as np
import pytest

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti

_usm_types_list = ["shared", "device", "host"]
_typestrs_list = [
    "b1",
    "u1",
    "i1",
    "u2",
    "i2",
    "u4",
    "i4",
    "u8",
    "i8",
    "f2",
    "f4",
    "f8",
    "c8",
    "c16",
]


@pytest.fixture(params=_usm_types_list)
def usm_type(request):
    return request.param


@pytest.fixture(params=_typestrs_list)
def src_typestr(request):
    return request.param


@pytest.fixture(params=_typestrs_list)
def dst_typestr(request):
    return request.param


def test_copy1d(src_typestr, dst_typestr):
    q = dpctl.SyclQueue()
    src_dt = np.dtype(src_typestr)
    if np.issubdtype(src_dt, np.integer):
        Xnp = np.random.randint(0, 2, size=4096).astype(src_dt)
    elif np.issubdtype(src_dt, np.floating):
        Xnp = np.random.randn(4096).astype(src_dt)
    elif np.issubdtype(src_dt, np.complexfloating):
        Xnp = np.random.randn(4096) + 1j * np.random.randn(4096)
        Xnp = Xnp.astype(src_dt)
    else:
        Xnp = np.random.randint(0, 2, size=4096).astype(src_dt)

    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.empty(Xnp.shape, dtype=dst_typestr, sycl_queue=q)
    ev = ti._copy_usm_ndarray_into_usm_ndarray(src=X, dst=Y, queue=q)
    ev.wait()
    Ynp = Xnp.astype(dst_typestr, casting="unsafe", copy=True)
    assert np.allclose(Ynp, dpt.asnumpy(Y))
    q.wait()
