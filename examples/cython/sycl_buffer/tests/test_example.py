import numpy as np
import pytest
from syclbuffer import columnwise_total

import dpctl


def test_columnwise_total():
    x = np.array([[2, 3], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)

    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Could not create default-constructed queue")

    ref = x.sum(axis=0)
    res1 = columnwise_total(x)
    assert res1.shape == (2,)

    res2 = columnwise_total(x, queue=q)
    assert res2.shape == (2,)

    assert np.allclose(res1, ref)
    assert np.allclose(res2, ref)
