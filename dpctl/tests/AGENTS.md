# dpctl/tests/ - Test Suite

## Purpose

pytest-based test suite for dpctl SYCL runtime objects and USM memory functionality.

**Note:** The `dpctl/tensor` module has been removed; focus testing on SYCL objects and memory APIs.

## Key Files

| File | Purpose |
|------|---------|
| `conftest.py` | Fixtures and pytest configuration |
| `helper/_helper.py` | `get_queue_or_skip()` (queue creation helper) |

## Essential Helpers

From `helper/_helper.py`:
```python
get_queue_or_skip()           # Create queue or skip test
```

## Test Pattern (SYCL objects)

```python
def test_device_creation():
    q = get_queue_or_skip()
    dev = q.sycl_device
    # ... test device properties and methods

def test_usm_allocation():
    q = get_queue_or_skip()
    # Test USM memory allocation for supported data types
    # ...
```

## Critical Rules

1. **Device compatibility:** Not all devices support fp64/fp16 – verify support before testing float64/complex128 memory operations.
2. **Queue consistency:** Memory allocations and operations must use compatible queues.
