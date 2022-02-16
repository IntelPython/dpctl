import numpy as np
import pytest
from sycl_gemm import gemv

import dpctl
import dpctl.tensor as dpt


def test_gemv():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    Mnp, vnp = np.random.randn(5, 3), np.random.randn(3)
    r = dpt.empty((5,), dtype="d", sycl_queue=q)
    M = dpt.asarray(Mnp, sycl_queue=q)
    v = dpt.asarray(vnp, sycl_queue=q)
    ev = gemv(M.sycl_queue, M, v, r, [])
    ev.wait()
    rnp = dpt.asnumpy(r)
    assert np.allclose(rnp, Mnp @ vnp)
