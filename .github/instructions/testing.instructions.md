---
applyTo:
  - "dpctl/tests/**/*.py"
  - "**/test_*.py"
---

# Testing Instructions

See `dpctl/tests/AGENTS.md` for patterns.

## Essential helpers (from `helper/_helper.py`)
```python
get_queue_or_skip()           # Create queue or skip
skip_if_dtype_not_supported() # Skip if device lacks dtype
```

## Dtype/USM lists
Import from `elementwise/utils.py` - do not hardcode.

## Coverage
- All dtypes from `_all_dtypes`
- All USM types: device, shared, host
- Edge cases: empty, scalar, broadcast
