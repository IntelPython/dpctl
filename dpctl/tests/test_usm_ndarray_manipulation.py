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


import numpy as np
import pytest
from numpy.testing import assert_array_equal

import dpctl
import dpctl.tensor as dpt


def test_permute_dims_incorrect_type():
    X_list = list([[1, 2, 3], [4, 5, 6]])
    X_tuple = tuple(X_list)
    Xnp = np.array(X_list)

    pytest.raises(TypeError, dpt.permute_dims, X_list, (1, 0))
    pytest.raises(TypeError, dpt.permute_dims, X_tuple, (1, 0))
    pytest.raises(TypeError, dpt.permute_dims, Xnp, (1, 0))


def test_permute_dims_empty_array():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.empty((10, 0))
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.permute_dims(X, (1, 0))
    Ynp = np.transpose(Xnp, (1, 0))
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_permute_dims_0d_1d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp_0d = np.array(1, dtype="int64")
    X_0d = dpt.asarray(Xnp_0d, sycl_queue=q)
    Y_0d = dpt.permute_dims(X_0d, ())
    assert_array_equal(dpt.asnumpy(Y_0d), dpt.asnumpy(X_0d))

    Xnp_1d = np.random.randint(0, 2, size=6, dtype="int64")
    X_1d = dpt.asarray(Xnp_1d, sycl_queue=q)
    Y_1d = dpt.permute_dims(X_1d, (0))
    assert_array_equal(dpt.asnumpy(Y_1d), dpt.asnumpy(X_1d))

    pytest.raises(ValueError, dpt.permute_dims, X_1d, ())
    pytest.raises(np.AxisError, dpt.permute_dims, X_1d, (1))
    pytest.raises(ValueError, dpt.permute_dims, X_1d, (1, 0))
    pytest.raises(
        ValueError, dpt.permute_dims, dpt.reshape(X_1d, (2, 3)), (1, 1)
    )


@pytest.mark.parametrize("shapes", [(2, 2), (1, 4), (3, 3, 3), (4, 1, 3)])
def test_permute_dims_2d_3d(shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp_size = np.prod(shapes)

    Xnp = np.random.randint(0, 2, size=Xnp_size, dtype="int64").reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    X_ndim = X.ndim
    if X_ndim == 2:
        Y = dpt.permute_dims(X, (1, 0))
        Ynp = np.transpose(Xnp, (1, 0))
    elif X_ndim == 3:
        X = dpt.asarray(Xnp, sycl_queue=q)
        Y = dpt.permute_dims(X, (2, 0, 1))
        Ynp = np.transpose(Xnp, (2, 0, 1))
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_expand_dims_incorrect_type():
    X_list = list([1, 2, 3, 4, 5])
    X_tuple = tuple(X_list)
    Xnp = np.array(X_list)

    pytest.raises(TypeError, dpt.permute_dims, X_list, 1)
    pytest.raises(TypeError, dpt.permute_dims, X_tuple, 1)
    pytest.raises(TypeError, dpt.permute_dims, Xnp, 1)


def test_expand_dims_0d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.array(1, dtype="int64")
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.expand_dims(X, 0)
    Ynp = np.expand_dims(Xnp, 0)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.expand_dims(X, -1)
    Ynp = np.expand_dims(Xnp, -1)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(np.AxisError, dpt.expand_dims, X, 1)
    pytest.raises(np.AxisError, dpt.expand_dims, X, -2)


@pytest.mark.parametrize("shapes", [(3,), (3, 3), (3, 3, 3)])
def test_expand_dims_1d_3d(shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp_size = np.prod(shapes)

    Xnp = np.random.randint(0, 2, size=Xnp_size, dtype="int64").reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    shape_len = len(shapes)
    for axis in range(-shape_len - 1, shape_len):
        Y = dpt.expand_dims(X, axis)
        Ynp = np.expand_dims(Xnp, axis)
        assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(np.AxisError, dpt.expand_dims, X, shape_len + 1)
    pytest.raises(np.AxisError, dpt.expand_dims, X, -shape_len - 2)


@pytest.mark.parametrize(
    "axes", [(0, 1, 2), (0, -1, -2), (0, 3, 5), (0, -3, -5)]
)
def test_expand_dims_tuple(axes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.empty((3, 3, 3))
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.expand_dims(X, axes)
    Ynp = np.expand_dims(Xnp, axes)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_expand_dims_incorrect_tuple():

    X = dpt.empty((3, 3, 3), dtype="i4")
    pytest.raises(np.AxisError, dpt.expand_dims, X, (0, -6))
    pytest.raises(np.AxisError, dpt.expand_dims, X, (0, 5))

    pytest.raises(ValueError, dpt.expand_dims, X, (1, 1))


def test_squeeze_incorrect_type():
    X_list = list([1, 2, 3, 4, 5])
    X_tuple = tuple(X_list)
    Xnp = np.array(X_list)

    pytest.raises(TypeError, dpt.permute_dims, X_list, 1)
    pytest.raises(TypeError, dpt.permute_dims, X_tuple, 1)
    pytest.raises(TypeError, dpt.permute_dims, Xnp, 1)


def test_squeeze_0d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.array(1)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.squeeze(X)
    Ynp = Xnp.squeeze()
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.squeeze(X, 0)
    Ynp = Xnp.squeeze(0)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.squeeze(X, (0))
    Ynp = Xnp.squeeze((0))
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.squeeze(X, -1)
    Ynp = Xnp.squeeze(-1)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(np.AxisError, dpt.squeeze, X, 1)
    pytest.raises(np.AxisError, dpt.squeeze, X, -2)
    pytest.raises(np.AxisError, dpt.squeeze, X, (1))
    pytest.raises(np.AxisError, dpt.squeeze, X, (-2))
    pytest.raises(ValueError, dpt.squeeze, X, (0, 0))


@pytest.mark.parametrize(
    "shapes",
    [
        (0),
        (1),
        (1, 2),
        (2, 1),
        (1, 1),
        (2, 2),
        (1, 0),
        (0, 1),
        (1, 2, 1),
        (2, 1, 2),
        (2, 2, 2),
        (1, 1, 1),
        (1, 0, 1),
        (0, 1, 0),
    ],
)
def test_squeeze_without_axes(shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.empty(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.squeeze(X)
    Ynp = Xnp.squeeze()
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("axes", [0, 2, (0), (2), (0, 2)])
def test_squeeze_axes_arg(axes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.array([[[1], [2], [3]]])
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.squeeze(X, axes)
    Ynp = Xnp.squeeze(axes)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize("axes", [1, -2, (1), (-2), (0, 0), (1, 1)])
def test_squeeze_axes_arg_error(axes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.array([[[1], [2], [3]]])
    X = dpt.asarray(Xnp, sycl_queue=q)
    pytest.raises(ValueError, dpt.squeeze, X, axes)


@pytest.mark.parametrize(
    "data",
    [
        [np.array(0), (0,)],
        [np.array(0), (1,)],
        [np.array(0), (3,)],
        [np.ones(1), (1,)],
        [np.ones(1), (2,)],
        [np.ones(1), (1, 2, 3)],
        [np.arange(3), (3,)],
        [np.arange(3), (1, 3)],
        [np.arange(3), (2, 3)],
        [np.ones(0), 0],
        [np.ones(1), 1],
        [np.ones(1), 2],
        [np.ones(1), (0,)],
        [np.ones((1, 2)), (0, 2)],
        [np.ones((2, 1)), (2, 0)],
    ],
)
def test_broadcast_to_succeeds(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp, target_shape = data
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.broadcast_to(X, target_shape)
    Ynp = np.broadcast_to(Xnp, target_shape)
    assert_array_equal(dpt.asnumpy(Y), Ynp)


@pytest.mark.parametrize(
    "data",
    [
        [(0,), ()],
        [(1,), ()],
        [(3,), ()],
        [(3,), (1,)],
        [(3,), (2,)],
        [(3,), (4,)],
        [(1, 2), (2, 1)],
        [(1, 1), (1,)],
        [(1,), -1],
        [(1,), (-1,)],
        [(1, 2), (-1, 2)],
    ],
)
def test_broadcast_to_raises(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    orig_shape, target_shape = data
    Xnp = np.zeros(orig_shape)
    X = dpt.asarray(Xnp, sycl_queue=q)
    pytest.raises(ValueError, dpt.broadcast_to, X, target_shape)


def assert_broadcast_correct(input_shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    np_arrays = [np.zeros(s) for s in input_shapes]
    out_np_arrays = np.broadcast_arrays(*np_arrays)
    usm_arrays = [dpt.asarray(Xnp, sycl_queue=q) for Xnp in np_arrays]
    out_usm_arrays = dpt.broadcast_arrays(*usm_arrays)
    for Xnp, X in zip(out_np_arrays, out_usm_arrays):
        assert_array_equal(
            Xnp, dpt.asnumpy(X), err_msg=f"Failed for {input_shapes})"
        )


def assert_broadcast_arrays_raise(input_shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    usm_arrays = [dpt.asarray(np.zeros(s), sycl_queue=q) for s in input_shapes]
    pytest.raises(ValueError, dpt.broadcast_arrays, *usm_arrays)


def test_broadcast_arrays_same():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    Xnp = np.arange(10)
    Ynp = np.arange(10)
    res_Xnp, res_Ynp = np.broadcast_arrays(Xnp, Ynp)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.asarray(Ynp, sycl_queue=q)
    res_X, res_Y = dpt.broadcast_arrays(X, Y)
    assert_array_equal(res_Xnp, dpt.asnumpy(res_X))
    assert_array_equal(res_Ynp, dpt.asnumpy(res_Y))


def test_broadcast_arrays_one_off():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")
    Xnp = np.array([[1, 2, 3]])
    Ynp = np.array([[1], [2], [3]])
    res_Xnp, res_Ynp = np.broadcast_arrays(Xnp, Ynp)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.asarray(Ynp, sycl_queue=q)
    res_X, res_Y = dpt.broadcast_arrays(X, Y)
    assert_array_equal(res_Xnp, dpt.asnumpy(res_X))
    assert_array_equal(res_Ynp, dpt.asnumpy(res_Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (),
        (1,),
        (3,),
        (0, 1),
        (0, 3),
        (1, 0),
        (3, 0),
        (1, 3),
        (3, 1),
        (3, 3),
    ],
)
def test_broadcast_arrays_same_shapes(shapes):
    for shape in shapes:
        single_input_shapes = [shape]
        assert_broadcast_correct(single_input_shapes)
        double_input_shapes = [shape, shape]
        assert_broadcast_correct(double_input_shapes)
        triple_input_shapes = [shape, shape, shape]
        assert_broadcast_correct(triple_input_shapes)


@pytest.mark.parametrize(
    "shapes",
    [
        [[(1,), (3,)]],
        [[(1, 3), (3, 3)]],
        [[(3, 1), (3, 3)]],
        [[(1, 3), (3, 1)]],
        [[(1, 1), (3, 3)]],
        [[(1, 1), (1, 3)]],
        [[(1, 1), (3, 1)]],
        [[(1, 0), (0, 0)]],
        [[(0, 1), (0, 0)]],
        [[(1, 0), (0, 1)]],
        [[(1, 1), (0, 0)]],
        [[(1, 1), (1, 0)]],
        [[(1, 1), (0, 1)]],
    ],
)
def test_broadcast_arrays_same_len_shapes(shapes):
    # Check that two different input shapes of the same length, but some have
    # ones, broadcast to the correct shape.

    for input_shapes in shapes:
        assert_broadcast_correct(input_shapes)
        assert_broadcast_correct(input_shapes[::-1])


@pytest.mark.parametrize(
    "shapes",
    [
        [[(), (3,)]],
        [[(3,), (3, 3)]],
        [[(3,), (3, 1)]],
        [[(1,), (3, 3)]],
        [[(), (3, 3)]],
        [[(1, 1), (3,)]],
        [[(1,), (3, 1)]],
        [[(1,), (1, 3)]],
        [[(), (1, 3)]],
        [[(), (3, 1)]],
        [[(), (0,)]],
        [[(0,), (0, 0)]],
        [[(0,), (0, 1)]],
        [[(1,), (0, 0)]],
        [[(), (0, 0)]],
        [[(1, 1), (0,)]],
        [[(1,), (0, 1)]],
        [[(1,), (1, 0)]],
        [[(), (1, 0)]],
        [[(), (0, 1)]],
    ],
)
def test_broadcast_arrays_different_len_shapes(shapes):
    # Check that two different input shapes (of different lengths) broadcast
    # to the correct shape.

    for input_shapes in shapes:
        assert_broadcast_correct(input_shapes)
        assert_broadcast_correct(input_shapes[::-1])


@pytest.mark.parametrize(
    "shapes",
    [
        [[(3,), (4,)]],
        [[(2, 3), (2,)]],
        [[(3,), (3,), (4,)]],
        [[(1, 3, 4), (2, 3, 3)]],
    ],
)
def test_incompatible_shapes_raise_valueerror(shapes):
    for input_shapes in shapes:
        assert_broadcast_arrays_raise(input_shapes)
        assert_broadcast_arrays_raise(input_shapes[::-1])


def test_flip_axes_incorrect():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    X_np = np.ones((4, 4))
    X = dpt.asarray(X_np, sycl_queue=q)

    pytest.raises(np.AxisError, dpt.flip, dpt.asarray(np.ones(4)), axes=1)
    pytest.raises(np.AxisError, dpt.flip, X, axes=2)
    pytest.raises(np.AxisError, dpt.flip, X, axes=-3)
    pytest.raises(np.AxisError, dpt.flip, X, axes=(0, 3))


def test_flip_0d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.array(1, dtype="int64")
    X = dpt.asarray(Xnp, sycl_queue=q)
    Ynp = np.flip(Xnp)
    Y = dpt.flip(X)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    pytest.raises(np.AxisError, dpt.flip, X, 0)
    pytest.raises(np.AxisError, dpt.flip, X, 1)
    pytest.raises(np.AxisError, dpt.flip, X, -1)


def test_flip_1d():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.arange(6)
    X = dpt.asarray(Xnp, sycl_queue=q)

    for ax in range(-X.ndim, X.ndim):
        Ynp = np.flip(Xnp, ax)
        Y = dpt.flip(X, ax)
        assert_array_equal(Ynp, dpt.asnumpy(Y))

    Ynp = np.flip(Xnp, 0)
    Y = dpt.flip(X, 0)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (3, 2),
        (2, 3),
        (2, 2),
        (3, 3),
        (3, 2, 3),
        (2, 3, 2),
        (2, 2, 2),
        (3, 3, 3),
    ],
)
def test_flip_2d_3d(shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp_size = np.prod(shapes)
    Xnp = np.arange(Xnp_size).reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    for ax in range(-X.ndim, X.ndim):
        Y = dpt.flip(X, ax)
        Ynp = np.flip(Xnp, ax)
        assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (1,),
        (3,),
        (2, 3),
        (3, 2),
        (2, 2),
        (1, 2, 3),
        (2, 1, 3),
        (2, 3, 1),
        (3, 2, 1),
        (3, 3, 3),
    ],
)
def test_flip_default_axes(shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp_size = np.prod(shapes)
    Xnp = np.arange(Xnp_size).reshape(shapes)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.flip(X)
    Ynp = np.flip(Xnp)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "shapes",
    [
        (0),
        (1),
        (1, 1),
        (1, 0),
        (0, 1),
        (1, 1, 1),
        (1, 0, 1),
        (0, 1, 0),
    ],
)
def test_flip_empty_0_size_dim(shapes):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    X = dpt.empty(shapes, sycl_queue=q)
    dpt.flip(X)


@pytest.mark.parametrize(
    "data",
    [
        [(2, 3), (0, 1)],
        [(2, 3), (1, 0)],
        [(2, 3), ()],
        [(2, 1, 3), (0, 2)],
        [(3, 1, 2), (2, 0)],
        [(3, 3, 3), (2,)],
        [(1, 2, 3), [0, -2]],
        [(3, 1, 2), [-1, 0]],
        [(3, 3, 3), [-2, -1]],
    ],
)
def test_flip_multiple_axes(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    shape, axes = data
    Xnp_size = np.prod(shape)
    Xnp = np.arange(Xnp_size).reshape(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)
    Y = dpt.flip(X, axes)
    Ynp = np.flip(Xnp, axes)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_roll_empty():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.empty([])
    X = dpt.asarray(Xnp, sycl_queue=q)

    Y = dpt.roll(X, 1)
    Ynp = np.roll(Xnp, 1)
    assert_array_equal(Ynp, dpt.asnumpy(Y))
    pytest.raises(np.AxisError, dpt.roll, X, 1, 0)
    pytest.raises(np.AxisError, dpt.roll, X, 1, 1)


@pytest.mark.parametrize(
    "data",
    [
        [2, None],
        [-2, None],
        [2, 0],
        [-2, 0],
        [2, ()],
        [11, 0],
    ],
)
def test_roll_1d(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.arange(10)
    X = dpt.asarray(Xnp, sycl_queue=q)
    sh, ax = data

    Y = dpt.roll(X, sh, ax)
    Ynp = np.roll(Xnp, sh, ax)
    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Y = dpt.roll(X, sh, ax)
    Ynp = np.roll(Xnp, sh, ax)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "data",
    [
        [1, None],
        [1, 0],
        [1, 1],
        [1, ()],
        # Roll multiple axes at once
        [1, (0, 1)],
        [(1, 0), (0, 1)],
        [(-1, 0), (1, 0)],
        [(0, 1), (0, 1)],
        [(0, -1), (0, 1)],
        [(1, 1), (0, 1)],
        [(-1, -1), (0, 1)],
        # Roll the same axis multiple times.
        [1, (0, 0)],
        [1, (1, 1)],
        # Roll more than one turn in either direction.
        [6, 1],
        [-4, 1],
    ],
)
def test_roll_2d(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xnp = np.arange(10).reshape(2, 5)
    X = dpt.asarray(Xnp, sycl_queue=q)
    sh, ax = data

    Y = dpt.roll(X, sh, ax)
    Ynp = np.roll(Xnp, sh, ax)
    assert_array_equal(Ynp, dpt.asnumpy(Y))


def test_concat_incorrect_type():
    Xnp = np.ones((2, 2))
    pytest.raises(TypeError, dpt.concat)
    pytest.raises(TypeError, dpt.concat, [])
    pytest.raises(TypeError, dpt.concat, Xnp)
    pytest.raises(TypeError, dpt.concat, [Xnp, Xnp])


def test_concat_incorrect_queue():
    try:
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    X = dpt.ones((2, 2), sycl_queue=q1)
    Y = dpt.ones((2, 2), sycl_queue=q2)

    pytest.raises(ValueError, dpt.concat, [X, Y])


def test_concat_incorrect_dtype():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    X = dpt.ones((2, 2), dtype=np.int64, sycl_queue=q)
    Y = dpt.ones((2, 2), dtype=np.uint64, sycl_queue=q)

    pytest.raises(ValueError, dpt.concat, [X, Y])


def test_concat_incorrect_ndim():
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    X = dpt.ones((2, 2), sycl_queue=q)
    Y = dpt.ones((2, 2, 2), sycl_queue=q)

    pytest.raises(ValueError, dpt.concat, [X, Y])


@pytest.mark.parametrize(
    "data",
    [
        [(2, 2), (3, 3), 0],
        [(2, 2), (3, 3), 1],
        [(3, 2), (3, 3), 0],
        [(2, 3), (3, 3), 1],
    ],
)
def test_concat_incorrect_shape(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xshape, Yshape, axis = data

    X = dpt.ones(Xshape, sycl_queue=q)
    Y = dpt.ones(Yshape, sycl_queue=q)

    pytest.raises(ValueError, dpt.concat, [X, Y], axis)


@pytest.mark.parametrize(
    "data",
    [
        [(6,), 0],
        [(2, 3), 1],
        [(3, 2), -1],
        [(1, 6), 0],
        [(2, 1, 3), 2],
    ],
)
def test_concat_1array(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xshape, axis = data

    Xnp = np.arange(6).reshape(Xshape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.concatenate([Xnp], axis=axis)
    Y = dpt.concat([X], axis=axis)

    assert_array_equal(Ynp, dpt.asnumpy(Y))

    Ynp = np.concatenate((Xnp,), axis=axis)
    Y = dpt.concat((X,), axis=axis)

    assert_array_equal(Ynp, dpt.asnumpy(Y))


@pytest.mark.parametrize(
    "data",
    [
        [(1,), (1,), 0],
        [(0, 2), (2, 2), 0],
        [(2, 1), (2, 2), -1],
        [(2, 2, 2), (2, 1, 2), 1],
    ],
)
def test_concat_2arrays(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xshape, Yshape, axis = data

    Xnp = np.ones(Xshape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.zeros(Yshape)
    Y = dpt.asarray(Ynp, sycl_queue=q)

    Znp = np.concatenate([Xnp, Ynp], axis=axis)
    Z = dpt.concat([X, Y], axis=axis)

    assert_array_equal(Znp, dpt.asnumpy(Z))


@pytest.mark.parametrize(
    "data",
    [
        [(1,), (1,), (1,), 0],
        [(0, 2), (2, 2), (1, 2), 0],
        [(2, 1, 2), (2, 2, 2), (2, 4, 2), 1],
    ],
)
def test_concat_3arrays(data):
    try:
        q = dpctl.SyclQueue()
    except dpctl.SyclQueueCreationError:
        pytest.skip("Queue could not be created")

    Xshape, Yshape, Zshape, axis = data

    Xnp = np.ones(Xshape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.zeros(Yshape)
    Y = dpt.asarray(Ynp, sycl_queue=q)

    Znp = np.full(Zshape, 2.0)
    Z = dpt.asarray(Znp, sycl_queue=q)

    Rnp = np.concatenate([Xnp, Ynp, Znp], axis=axis)
    R = dpt.concat([X, Y, Z], axis=axis)

    assert_array_equal(Rnp, dpt.asnumpy(R))
