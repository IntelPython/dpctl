---
applyTo:
  - "dpctl/tensor/*.py"
  - "dpctl/tensor/**/*.py"
---

# Tensor Python Instructions

See `dpctl/tensor/AGENTS.md` for patterns.

## Queue validation (required)
```python
exec_q = dpctl.utils.get_execution_queue([x.sycl_queue, y.sycl_queue])
if exec_q is None:
    raise ExecutionPlacementError("...")
```

## Adding operations
See checklist in `dpctl/tensor/AGENTS.md`.
