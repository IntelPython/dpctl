import advanced
import numpy as np
import pytest

import dpctl.tensor as dpt


def test_basic_slice1():
    res = advanced._basic_slice_meta((0,), (1,), (1,), 0)
    assert res == (tuple(), tuple(), 0, tuple(), -1)


def test_basic_slice1a():
    res = advanced._basic_slice_meta(0, (1,), (1,), 0)
    assert res == (tuple(), tuple(), 0, tuple(), -1)


def test_basic_slice2():
    res = advanced._basic_slice_meta((slice(None),), (1,), (1,), 0)
    assert res == ((1,), (1,), 0, tuple(), -1)


def test_basic_slice3():
    res = advanced._basic_slice_meta((slice(None, None, -1),), (1,), (1,), 0)
    assert res == ((1,), (-1,), 0, tuple(), -1)


def test_basic_slice4():
    res = advanced._basic_slice_meta(
        (slice(None, None, -1),),
        (
            5,
            3,
        ),
        (
            3,
            1,
        ),
        0,
    )
    assert res == ((5, 3), (-3, 1), (5 - 1) * 3, tuple(), -1)


def test_basic_slice5():
    res = advanced._basic_slice_meta(
        (
            slice(None),
            slice(None, None, -1),
        ),
        (
            4,
            3,
        ),
        (
            3,
            1,
        ),
        0,
    )
    assert res == ((4, 3), (3, -1), 3 - 1, tuple(), -1)


def test_basic_slice6():
    res = advanced._basic_slice_meta(
        (
            2,
            slice(None, None, -1),
        ),
        (
            4,
            3,
        ),
        (
            3,
            1,
        ),
        0,
    )
    assert res == ((3,), (-1,), 2 * 3 + 3 - 1, tuple(), -1)


def test_basic_slice7():
    res = advanced._basic_slice_meta(
        (
            Ellipsis,
            slice(None, None, -1),
        ),
        (
            4,
            3,
        ),
        (
            3,
            1,
        ),
        0,
    )
    assert res == ((4, 3), (3, -1), 3 - 1, tuple(), -1)


def test_basic_slice8():
    res = advanced._basic_slice_meta(
        (Ellipsis, None),
        (
            4,
            3,
        ),
        (
            3,
            1,
        ),
        0,
    )
    assert res == ((4, 3, 1), (3, 1, 0), 0, tuple(), -1)


def test_basic_slice9():
    res = advanced._basic_slice_meta(
        (
            None,
            Ellipsis,
        ),
        (
            4,
            3,
        ),
        (
            3,
            1,
        ),
        0,
    )
    assert res == (
        (
            1,
            4,
            3,
        ),
        (0, 3, 1),
        0,
        tuple(),
        -1,
    )


def test_basic_slice10():
    res = advanced._basic_slice_meta(
        (None, Ellipsis, slice(None)), (4, 3, 5), (30, 5, 1), 0
    )
    assert res == ((1, 4, 3, 5), (0, 30, 5, 1), 0, tuple(), -1)


def test_advanced_slice1():
    ii = dpt.asarray([0, 1])
    res = advanced._basic_slice_meta((ii,), (10,), (1,), 0)
    assert res == ((10,), (1,), 0, (ii,), 0)

    res = advanced._basic_slice_meta(ii, (10,), (1,), 0)
    assert res == ((10,), (1,), 0, (ii,), 0)


def test_advanced_slice2():
    ii = dpt.asarray([0, 1])
    res = advanced._basic_slice_meta((ii, None), (10,), (1,), 0)
    assert res == ((10, 1), (1, 0), 0, (ii,), 0)


def test_advanced_slice3():
    ii = dpt.asarray([0, 1])
    res = advanced._basic_slice_meta((None, ii), (10,), (1,), 0)
    assert res == (
        (
            1,
            10,
        ),
        (
            0,
            1,
        ),
        0,
        (ii,),
        1,
    )


def test_advanced_slice4():
    ii = dpt.asarray([0, 1])
    res = advanced._basic_slice_meta(
        (ii, ii, ii),
        (10, 10, 10),
        (
            100,
            10,
            1,
        ),
        0,
    )
    assert res == (
        (10, 10, 10),
        (
            100,
            10,
            1,
        ),
        0,
        (ii, ii, ii),
        0,
    )


def test_advanced_slice5():
    ii = dpt.asarray([0, 1])
    with pytest.raises(IndexError):
        advanced._basic_slice_meta(
            (ii, slice(None), ii),
            (10, 10, 10),
            (
                100,
                10,
                1,
            ),
            0,
        )


def test_advanced_slice6():
    ii = dpt.asarray([0, 1])
    res = advanced._basic_slice_meta(
        (
            slice(None),
            ii,
            ii,
        ),
        (10, 10, 10),
        (
            100,
            10,
            1,
        ),
        0,
    )
    assert res == (
        (
            10,
            10,
            10,
        ),
        (100, 10, 1),
        0,
        (
            ii,
            ii,
        ),
        1,
    )


def test_advanced_slice7():
    x = dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype="i8"),
        (
            3,
            3,
            3,
        ),
    )
    mask = dpt.asarray(
        [
            [[True, True, False], [False, True, True], [True, False, True]],
            [[True, False, False], [False, False, True], [False, True, False]],
            [[True, True, True], [False, False, False], [False, False, True]],
        ]
    )
    res = advanced.get_item(x, mask)
    res_expected = np.array([0, 1, 4, 5, 6, 8, 9, 14, 16, 18, 19, 20, 26])
    assert np.array_equal(dpt.asnumpy(res), res_expected)
    res2 = advanced.get_item(x, (mask,))
    assert np.array_equal(dpt.asnumpy(res2), res_expected)


def test_advanced_slice8():
    x = dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype="i8"),
        (
            3,
            3,
            3,
        ),
    )
    mask = dpt.asarray(
        [[True, False, False], [False, True, False], [False, True, False]]
    )
    res = advanced.get_item(x, mask)
    res_expected = np.array([[0, 1, 2], [12, 13, 14], [21, 22, 23]])
    assert np.array_equal(dpt.asnumpy(res), res_expected)
    res2 = advanced.get_item(x, (mask,))
    assert np.array_equal(dpt.asnumpy(res2), res_expected)


def test_advanced_slice9():
    x = dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype="i8"),
        (
            3,
            3,
            3,
        ),
    )
    mask = dpt.asarray(
        [[True, False, False], [False, True, False], [False, True, False]]
    )
    res = advanced.get_item(
        x,
        (
            slice(None, None, None),
            mask,
        ),
    )
    res_expected = np.array([[0, 4, 7], [9, 13, 16], [18, 22, 25]])
    assert np.array_equal(dpt.asnumpy(res), res_expected)


def lin_id(i, j, k):
    return 9 * i + 3 * j + k


def test_advanced_slice10():
    x = dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype="i8"),
        (
            3,
            3,
            3,
        ),
    )
    i0 = dpt.asarray([0, 1, 1])
    i1 = dpt.asarray([1, 1, 2])
    i2 = dpt.asarray([2, 0, 1])
    res = advanced.get_item(x, (i0, i1, i2))
    res_expected = np.array(
        [
            lin_id(0, 1, 2),
            lin_id(1, 1, 0),
            lin_id(1, 2, 1),
        ]
    )
    assert np.array_equal(dpt.asnumpy(res), res_expected)


def test_advanced_slice11():
    x = dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype="i8"),
        (
            3,
            3,
            3,
        ),
    )
    i0 = dpt.asarray([0, 1, 1])
    i2 = dpt.asarray([2, 0, 1])
    with pytest.raises(IndexError):
        advanced.get_item(x, (i0, slice(None, None, None), i2))


def test_advanced_slice12():
    x = dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype="i8"),
        (
            3,
            3,
            3,
        ),
    )
    i1 = dpt.asarray([1, 1, 2])
    i2 = dpt.asarray([2, 0, 1])
    res = advanced.get_item(x, (slice(None), None, i1, i2, None))
    res_expected = np.array(
        [
            [[[lin_id(0, 1, 2)], [lin_id(0, 1, 0)], [lin_id(0, 2, 1)]]],
            [[[lin_id(1, 1, 2)], [lin_id(1, 1, 0)], [lin_id(1, 2, 1)]]],
            [[[lin_id(2, 1, 2)], [lin_id(2, 1, 0)], [lin_id(2, 2, 1)]]],
        ]
    )
    assert np.array_equal(dpt.asnumpy(res), res_expected)


def test_advanced_slice13():
    x = dpt.reshape(
        dpt.arange(3 * 3 * 3, dtype="i8"),
        (
            3,
            3,
            3,
        ),
    )
    i1 = dpt.asarray([[1], [2]])
    i2 = dpt.asarray([[0, 1]])
    res = advanced.get_item(x, (i1, i2, 0))
    res_expected = np.array(
        [
            [lin_id(1, 0, 0), lin_id(1, 1, 0)],
            [lin_id(2, 0, 0), lin_id(2, 1, 0)],
        ]
    )
    assert np.array_equal(dpt.asnumpy(res), res_expected)
