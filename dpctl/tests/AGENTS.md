# dpctl/tests/ - Test Suite

## Purpose

pytest-based test suite for dpctl functionality.

## Key Files

| File | Purpose |
|------|---------|
| `conftest.py` | Fixtures and pytest configuration |
| `helper/_helper.py` | `get_queue_or_skip()`, `skip_if_dtype_not_supported()` |

## Essential Helpers

From `helper/_helper.py`:
```python
get_queue_or_skip()           # Create queue or skip test
skip_if_dtype_not_supported() # Skip if device lacks dtype (fp64/fp16)
```

## Test Pattern

```python
@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "c8", "c16", ...])
def test_operation(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)
    # ... test logic and assertions
```

## Coverage Requirements

- All supported dtypes
- All USM types: device, shared, host
- Memory orders: C, F where applicable
- Edge cases: empty arrays, 0-d arrays (scalars), broadcasting

## Critical Rules

1. **Device compatibility:** Not all devices support fp64/fp16 - never assume availability
2. **Queue consistency:** Arrays in same operation must share compatible queues
