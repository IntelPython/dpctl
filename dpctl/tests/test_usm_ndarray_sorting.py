import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_sort_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    inp = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )

    s = dpt.sort(inp, descending=False)
    assert dpt.all(s[:-1] <= s[1:])

    s1 = dpt.sort(inp, descending=True)
    assert dpt.all(s1[:-1] >= s1[1:])


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_sort_2d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    fl = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )
    inp = dpt.reshape(fl, (20, -1))

    s = dpt.sort(inp, axis=1, descending=False)
    assert dpt.all(s[:, :-1] <= s[:, 1:])

    s1 = dpt.sort(inp, axis=1, descending=True)
    assert dpt.all(s1[:, :-1] >= s1[:, 1:])


def test_sort_strides():

    fl = dpt.roll(
        dpt.concat((dpt.ones(10000, dtype="i4"), dpt.zeros(10000, dtype="i4"))),
        734,
    )
    inp = dpt.reshape(fl, (-1, 20))

    s = dpt.sort(inp, axis=0, descending=False)
    assert dpt.all(s[:-1, :] <= s[1:, :])

    s1 = dpt.sort(inp, axis=0, descending=True)
    assert dpt.all(s1[:-1, :] >= s1[1:, :])


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_argsort_1d(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    inp = dpt.roll(
        dpt.concat(
            (dpt.ones(10000, dtype=dtype), dpt.zeros(10000, dtype=dtype))
        ),
        734,
    )

    s_idx = dpt.argsort(inp, descending=False)
    assert dpt.all(inp[s_idx[:-1]] <= inp[s_idx[1:]])

    s1_idx = dpt.argsort(inp, descending=True)
    assert dpt.all(inp[s1_idx[:-1]] >= inp[s1_idx[1:]])


def test_sort_validation():
    with pytest.raises(TypeError):
        dpt.sort(dict())


def test_argsort_validation():
    with pytest.raises(TypeError):
        dpt.argsort(dict())


def test_sort_axis0():
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    s = dpt.sort(x, axis=0)

    assert dpt.all(s[:-1, :] <= s[1:, :])


def test_argsort_axis0():
    get_queue_or_skip()

    n, m = 200, 30
    xf = dpt.arange(n * m, 0, step=-1, dtype="i4")
    x = dpt.reshape(xf, (n, m))
    idx = dpt.argsort(x, axis=0)

    s = x[idx, dpt.arange(m)[dpt.newaxis, :]]

    assert dpt.all(s[:-1, :] <= s[1:, :])


def test_sort_strided():
    get_queue_or_skip()

    x_orig = dpt.arange(100, dtype="i4")
    x_flipped = dpt.flip(x_orig, axis=0)
    s = dpt.sort(x_flipped)

    assert dpt.all(s == x_orig)


def test_argsort_strided():
    get_queue_or_skip()

    x_orig = dpt.arange(100, dtype="i4")
    x_flipped = dpt.flip(x_orig, axis=0)
    idx = dpt.argsort(x_flipped)

    assert dpt.all(x_flipped[idx] == x_orig)
