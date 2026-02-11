# dpctl/tests/ - Test Suite

## Purpose

pytest-based test suite for dpctl functionality.

## Key Files

| File | Purpose |
|------|---------|
| `conftest.py` | Fixtures and pytest configuration |
| `helper/_helper.py` | `get_queue_or_skip()`, `skip_if_dtype_not_supported()` |
| `elementwise/utils.py` | Dtype and USM type lists for parametrization |

## Essential Helpers

From `helper/_helper.py`:
```python
get_queue_or_skip()           # Create queue or skip test
skip_if_dtype_not_supported() # Skip if device lacks dtype (fp64/fp16)
```

## Dtype/USM Lists

**Do not hardcode** - import from `elementwise/utils.py`:
```python
from .utils import _all_dtypes, _usm_types, _no_complex_dtypes
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

- All supported dtypes (see `elementwise/utils.py`)
- All USM types: device, shared, host
- Memory orders: C, F where applicable
- Edge cases: empty arrays, 0-d arrays (scalars), broadcasting
