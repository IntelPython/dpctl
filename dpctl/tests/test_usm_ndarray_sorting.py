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
