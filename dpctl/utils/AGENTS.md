# dpctl/utils/ - Utility Functions

## Purpose

Helper utilities for device queries, execution context management, and ordering.

## Key Files

| File | Purpose |
|------|---------|
| `_compute_follows_data.pyx` | `get_execution_queue()` for queue compatibility |
| `_order_manager.py` | `OrderManager` for memory layout handling |
| `_intel_device_info.py` | Intel-specific device information |
| `_onetrace_context.py` | Tracing/profiling context manager |

## Key Functions

### get_execution_queue()
Validates queue compatibility between arrays:
```python
from dpctl.utils import get_execution_queue

exec_q = get_execution_queue([x.sycl_queue, y.sycl_queue])
if exec_q is None:
    raise ExecutionPlacementError("Incompatible queues")
```

### ExecutionPlacementError
Exception raised when arrays are on incompatible queues.

## Notes

- `get_execution_queue()` is critical for all tensor operations
- See usage examples in `dpctl/tensor/` modules
