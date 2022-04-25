import external_usm_allocation as eua
import numpy as np

import dpctl
import dpctl.memory as dpm


def test_dmatrix():
    q = dpctl.SyclQueue()
    matr = eua.DMatrix(q, 5, 5)
    assert hasattr(matr, "__sycl_usm_array_interface__")

    blob = dpm.as_usm_memory(matr)
    assert blob.get_usm_type() == "shared"

    Xh = np.array(
        [
            [1, 1, 1, 2, 2],
            [1, 0, 1, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, -1],
            [0, 0, 0, -1, 5],
        ],
        dtype="d",
    )
    host_bytes_view = Xh.reshape((-1)).view(np.ubyte)
    blob.copy_from_host(host_bytes_view)

    list_of_lists = matr.tolist()
    assert list_of_lists == Xh.tolist()
