#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers

import numpy as np
import pytest

import dpctl

# import dpctl.memory as dpmem
import dpctl.tensor as dpt


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (4,),
        (0,),
        (0, 1),
        (0, 0),
        (4, 5),
        (2, 5, 2),
        (2, 2, 2, 2, 2, 2, 2, 2),
    ],
)
@pytest.mark.parametrize("usm_type", ["shared", "host", "device"])
def test_allocate_usm_ndarray(shape, usm_type):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclCreationError:
        pytest.skip("Default SYCL queue could not be created")
    X = dpt.usm_ndarray(
        shape, dtype="d", buffer=usm_type, buffer_ctor_kwargs={"queue": q}
    )
    Xnp = np.ndarray(shape, dtype="d")
    assert X.usm_type == usm_type
    assert X.sycl_context == q.sycl_context
    assert X.sycl_device == q.sycl_device
    assert X.size == Xnp.size
    assert X.shape == Xnp.shape


@pytest.mark.parametrize(
    "dtype",
    [
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
    ],
)
def test_dtypes(dtype):
    dpt.usm_ndarray((1,), dtype=dtype)


def test_properties():
    """
    Test that properties execute
    """
    X = dpt.usm_ndarray((3, 4, 5), dtype="c16")
    assert isinstance(X.sycl_queue, dpctl.SyclQueue)
    assert isinstance(X.sycl_device, dpctl.SyclDevice)
    assert isinstance(X.sycl_context, dpctl.SyclContext)
    assert isinstance(X.dtype, np.dtype)
    assert isinstance(X.__sycl_usm_array_interface__, dict)
    assert isinstance(X.T, dpt.usm_ndarray)
    assert isinstance(X.imag, dpt.usm_ndarray)
    assert isinstance(X.real, dpt.usm_ndarray)
    assert isinstance(X.shape, tuple)
    assert isinstance(X.strides, tuple)
    assert X.usm_type in ("shared", "device", "host")
    assert isinstance(X.size, numbers.Integral)
    assert isinstance(X.nbytes, numbers.Integral)
    assert isinstance(X.ndim, numbers.Integral)


@pytest.mark.parametrize(
    "ind",
    [
        tuple(),
        (None,),
        (
            None,
            Ellipsis,
            None,
        ),
        (2, 2, None, 3, 4),
        (Ellipsis,),
        (None, slice(0, None, 2), Ellipsis, slice(0, None, 3)),
        (
            slice(None, None, -1),
            slice(None, None, -1),
            slice(0, None, 3),
            slice(1, None, 2),
        ),
    ],
)
def test_basic_slice(ind):
    X = dpt.usm_ndarray((2 * 3, 2 * 4, 3 * 5, 3 * 7), dtype="u1")
    Xnp = np.empty(X.shape, dtype=X.dtype)
    S = X[ind]
    Snp = Xnp[ind]
    assert S.shape == Snp.shape
    assert S.strides == Snp.strides
    assert S.dtype == X.dtype
