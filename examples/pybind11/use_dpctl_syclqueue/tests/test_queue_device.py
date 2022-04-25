import numpy as np
import use_queue_device as uqd

import dpctl


def test_compute_units():
    q = dpctl.SyclQueue()
    mcu = uqd.get_max_compute_units(q)

    assert type(mcu) is int
    assert mcu == q.sycl_device.max_compute_units


def test_global_memory():
    d = dpctl.SyclDevice()
    gm = uqd.get_device_global_mem_size(d)
    assert type(gm) is int
    assert gm == d.global_mem_size


def test_local_memory():
    d = dpctl.SyclDevice()
    lm = uqd.get_device_local_mem_size(d)
    assert type(lm) is int
    assert lm == d.local_mem_size


def test_offload_array_mod():
    execution_queue = dpctl.SyclQueue()
    X = np.random.randint(low=1, high=2**16 - 1, size=10**6, dtype=np.int64)
    modulus_p = 347

    # Y is a regular NumPy array with NumPy allocated host memory
    Y = uqd.offloaded_array_mod(execution_queue, X, modulus_p)

    Ynp = X % modulus_p

    assert np.array_equal(Y, Ynp)
