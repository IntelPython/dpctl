# dpctl/tests/ - Test Suite

## Purpose

pytest-based test suite for dpctl functionality.

## Key Files

| File | Purpose |
|------|---------|
| `conftest.py` | Fixtures and pytest configuration |
| `helper/_helper.py` | Test utilities |
| `elementwise/utils.py` | Dtype lists for parametrization |

## Essential Fixtures

From `helper/_helper.py`:
```python
get_queue_or_skip()           # Create queue or skip test
skip_if_dtype_not_supported() # Skip if device lacks dtype
```

## Standard Dtype Lists

Defined in `elementwise/utils.py`:
```python
_integral_dtypes = ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]
_real_fp_dtypes = ["f2", "f4", "f8"]
_complex_fp_dtypes = ["c8", "c16"]
_usm_types = ["device", "shared", "host"]
```

## Test Pattern

```python
@pytest.mark.parametrize("dtype", _all_dtypes)
def test_operation(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.ones(100, dtype=dtype, sycl_queue=q)
    result = dpt.operation(x)
    # ... assertions
```

## Coverage Requirements

- All supported dtypes
- All USM types (device, shared, host)
- Memory orders (C, F) where applicable
- Edge cases: empty arrays, scalars, broadcasting
