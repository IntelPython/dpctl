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
```

## Focus
Test SYCL runtime objects (Device, Queue, Context, Event, Platform) and USM memory APIs.

**Note:** The `dpctl/tensor` module has been removed; do not reference tensor‑specific patterns.
