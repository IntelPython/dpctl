---
applyTo:
  - "dpctl/tests/**/*.py"
  - "**/test_*.py"
---

# Testing Instructions

See [dpctl/tests/AGENTS.md](/dpctl/tests/AGENTS.md) for patterns and dtype lists.

## Quick Reference

### Essential helpers (from `helper/_helper.py`)
```python
get_queue_or_skip()           # Create queue or skip
skip_if_dtype_not_supported() # Skip if device lacks dtype
```

### Dtype lists (from `elementwise/utils.py`)
Use `_all_dtypes`, `_usm_types` for parametrization.

### Coverage requirements
- All supported dtypes
- All USM types (device, shared, host)
- Edge cases: empty arrays, scalars
