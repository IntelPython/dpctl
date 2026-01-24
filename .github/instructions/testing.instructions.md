---
applyTo:
  - "dpctl/tests/**/*.py"
  - "**/test_*.py"
  - "**/*_test.py"
---

# Testing Instructions

## Context

DPCTL uses pytest for testing. Tests must handle device availability, dtype support variations, and multiple memory configurations.

## Test Location

```
dpctl/tests/
├── conftest.py              # Fixtures and pytest configuration
├── helper/                  # Test helper utilities
│   └── _helper.py
├── elementwise/             # Elementwise operation tests
│   ├── utils.py             # Shared dtype lists
│   └── test_*.py
├── test_sycl_*.py           # Core SYCL object tests
├── test_tensor_*.py         # Tensor operation tests
└── test_usm_ndarray_*.py    # Array tests
```

## Essential Fixtures and Helpers

### Device/Queue Creation

```python
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

def test_something():
    # Skip test if no device available
    q = get_queue_or_skip()

    # Skip if specific device type needed
    q = get_queue_or_skip(("gpu",))
```

### Dtype Support Check

```python
def test_with_dtype():
    q = get_queue_or_skip()
    dtype = "f8"  # float64

    # Skip if device doesn't support this dtype
    skip_if_dtype_not_supported(dtype, q)

    # Test implementation
    x = dpt.ones(10, dtype=dtype, sycl_queue=q)
```

## Standard Dtype Lists

From `dpctl/tests/elementwise/utils.py`:

```python
# Integer types (always supported)
_integral_dtypes = ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]

# Floating point types (may need device support check)
_real_fp_dtypes = ["f2", "f4", "f8"]  # f8 (float64) needs fp64 support

# Complex types
_complex_fp_dtypes = ["c8", "c16"]  # c16 needs fp64 support

# All numeric types
_all_dtypes = _integral_dtypes + _real_fp_dtypes + _complex_fp_dtypes

# Boolean
_boolean_dtypes = ["?"]

# USM allocation types
_usm_types = ["device", "shared", "host"]

# Memory orders
_orders = ["C", "F", "A", "K"]
```

## Test Patterns

### Basic Parametrized Test

```python
import pytest
import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

@pytest.mark.parametrize("dtype", ["f4", "f8", "c8", "c16"])
def test_operation(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.linspace(0, 1, 100, dtype=dtype, sycl_queue=q)
    result = dpt.some_operation(x)

    expected = np.some_operation(dpt.asnumpy(x))
    assert_allclose(dpt.asnumpy(result), expected)
```

### Dtype Matrix Test (Binary Operations)

```python
@pytest.mark.parametrize("dtype1", _all_dtypes)
@pytest.mark.parametrize("dtype2", _all_dtypes)
def test_binary_dtype_matrix(dtype1, dtype2):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype1, q)
    skip_if_dtype_not_supported(dtype2, q)

    x = dpt.ones(10, dtype=dtype1, sycl_queue=q)
    y = dpt.ones(10, dtype=dtype2, sycl_queue=q)

    result = dpt.add(x, y)

    # Verify result dtype matches NumPy promotion
    expected_dtype = np.result_type(np.dtype(dtype1), np.dtype(dtype2))
    assert result.dtype == expected_dtype
```

### USM Type Test

```python
@pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
def test_usm_types(usm_type):
    q = get_queue_or_skip()

    x = dpt.ones(10, dtype="f4", usm_type=usm_type, sycl_queue=q)
    result = dpt.abs(x)

    assert result.usm_type == usm_type
```

### Memory Order Test

```python
@pytest.mark.parametrize("order", ["C", "F"])
def test_memory_order(order):
    q = get_queue_or_skip()

    x = dpt.ones((10, 10), dtype="f4", order=order, sycl_queue=q)

    if order == "C":
        assert x.flags.c_contiguous
    else:
        assert x.flags.f_contiguous
```

### Edge Case Tests

```python
def test_empty_array():
    q = get_queue_or_skip()
    x = dpt.empty((0,), dtype="f4", sycl_queue=q)
    result = dpt.abs(x)
    assert result.shape == (0,)

def test_scalar():
    q = get_queue_or_skip()
    x = dpt.asarray(5.0, dtype="f4", sycl_queue=q)
    result = dpt.abs(x)
    assert result.ndim == 0

def test_broadcast():
    q = get_queue_or_skip()
    x = dpt.ones((10, 1), dtype="f4", sycl_queue=q)
    y = dpt.ones((1, 10), dtype="f4", sycl_queue=q)
    result = dpt.add(x, y)
    assert result.shape == (10, 10)
```

## Test Naming Convention

```python
# Function being tested: dpt.abs()

def test_abs_basic():
    """Basic functionality test"""

def test_abs_dtype_matrix():
    """Test all dtype combinations"""

def test_abs_usm_types():
    """Test all USM allocation types"""

def test_abs_order():
    """Test memory order preservation"""

def test_abs_empty():
    """Test with empty array"""

def test_abs_scalar():
    """Test with 0-d array"""

def test_abs_broadcast():
    """Test broadcasting behavior"""

def test_abs_out_param():
    """Test output array parameter"""

def test_abs_inplace():
    """Test in-place operation (if supported)"""
```

## Assertions

```python
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# For exact integer comparisons
assert_array_equal(dpt.asnumpy(result), expected)

# For floating point with tolerance
assert_allclose(dpt.asnumpy(result), expected, rtol=1e-5, atol=1e-8)

# For dtype checking
assert result.dtype == dpt.float32

# For shape checking
assert result.shape == (10, 10)

# For USM type
assert result.usm_type == "device"

# For queue equality
assert result.sycl_queue == x.sycl_queue
```

## Conftest Markers

```python
# Mark test as having known complex number issues
@pytest.mark.broken_complex

# Skip on CI
@pytest.mark.skip(reason="Known issue #123")

# Expected failure
@pytest.mark.xfail(reason="Not yet implemented")
```

## Best Practices

1. **Always use fixtures** - Use `get_queue_or_skip()` instead of direct queue creation
2. **Check dtype support** - Use `skip_if_dtype_not_supported()` for fp64/fp16
3. **Test all dtypes** - Use parametrization to cover dtype matrix
4. **Test USM types** - Cover device, shared, and host memory
5. **Test edge cases** - Empty arrays, scalars, broadcasts
6. **Compare with NumPy** - Verify results match NumPy behavior
7. **Clean up resources** - Let arrays go out of scope naturally
8. **Use descriptive names** - Test names should indicate what's being tested
