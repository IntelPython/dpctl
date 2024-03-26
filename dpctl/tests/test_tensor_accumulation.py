import pytest
from helper import get_queue_or_skip

import dpctl.tensor as dpt

sint_types = [
    dpt.int8,
    dpt.int16,
    dpt.int32,
    dpt.int64,
]
uint_types = [
    dpt.uint8,
    dpt.uint16,
    dpt.uint32,
    dpt.uint64,
]
rfp_types = [
    dpt.float16,
    dpt.float32,
    dpt.float64,
]
cfp_types = [
    dpt.complex64,
    dpt.complex128,
]

all_types = [dpt.bool] + sint_types + uint_types + rfp_types + cfp_types


@pytest.mark.parametrize("dt", sint_types)
def test_contig_cumsum_sint(dt):
    get_queue_or_skip()
    n = 10000
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), n)

    res = dpt.cumulative_sum(x, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)


@pytest.mark.parametrize("dt", sint_types)
def test_strided_cumsum_sint(dt):
    get_queue_or_skip()
    n = 10000
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), 2 * n)[1::2]

    res = dpt.cumulative_sum(x, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)

    x2 = dpt.repeat(dpt.asarray([-1, 1], dtype=dt), 2 * n)[-1::-2]

    res = dpt.cumulative_sum(x2, dtype=dt)

    ar = dpt.arange(n, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == expected)


@pytest.mark.parametrize("dt", sint_types)
def test_contig_cumsum_axis_sint(dt):
    get_queue_or_skip()
    n0, n1 = 1000, 173
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), n0)
    m = dpt.tile(dpt.expand_dims(x, axis=1), (1, n1))

    res = dpt.cumulative_sum(m, dtype=dt, axis=0)

    ar = dpt.arange(n0, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == dpt.expand_dims(expected, axis=1))


@pytest.mark.parametrize("dt", sint_types)
def test_strided_cumsum_axis_sint(dt):
    get_queue_or_skip()
    n0, n1 = 1000, 173
    x = dpt.repeat(dpt.asarray([1, -1], dtype=dt), 2 * n0)
    m = dpt.tile(dpt.expand_dims(x, axis=1), (1, n1))[1::2, ::-1]

    res = dpt.cumulative_sum(m, dtype=dt, axis=0)

    ar = dpt.arange(n0, dtype=dt)
    expected = dpt.concat((1 + ar, dpt.flip(ar)))
    assert dpt.all(res == dpt.expand_dims(expected, axis=1))


def test_accumulate_scalar():
    get_queue_or_skip()

    s = dpt.asarray(1, dtype="i8")
    r = dpt.cumulative_sum(s)
    assert r == s
    assert r.ndim == s.ndim

    r = dpt.cumulative_sum(s, include_initial=True)
    r_expected = dpt.asarray([0, 1], dtype="i8")
    assert dpt.all(r == r_expected)
