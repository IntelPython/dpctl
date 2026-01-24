# dpctl/tensor/ - Array API Tensor Operations

## Purpose

Python Array API-compliant tensor operations using `usm_ndarray` on SYCL devices.

## Key Files

| File | Purpose |
|------|---------|
| `_usmarray.pyx` | `usm_ndarray` extension type |
| `_elementwise_funcs.py` | Elementwise operation wrappers |
| `_elementwise_common.py` | `UnaryElementwiseFunc`, `BinaryElementwiseFunc` |
| `_reduction.py` | Reduction operations (sum, prod, etc.) |
| `_manipulation_functions.py` | reshape, concat, stack, etc. |
| `_ctors.py` | Array constructors (empty, zeros, ones) |
| `_type_utils.py` | Type promotion and validation |

See [libtensor/AGENTS.md](libtensor/AGENTS.md) for C++ kernel implementation.

## Elementwise Wrapper Pattern

```python
from ._elementwise_common import UnaryElementwiseFunc
import dpctl.tensor._tensor_impl as ti

abs = UnaryElementwiseFunc(
    "abs",                # Operation name
    ti._abs_result_type,  # Type inference
    ti._abs,              # Kernel implementation
    _abs_docstring_
)
```

## Queue Validation

All operations must validate queue compatibility:

```python
exec_q = dpctl.utils.get_execution_queue([x.sycl_queue, y.sycl_queue])
if exec_q is None:
    raise ExecutionPlacementError("Arrays on incompatible queues")
```

## Adding New Operations

1. C++ kernel in `libtensor/include/kernels/`
2. C++ source in `libtensor/source/`
3. Register in `libtensor/source/tensor_elementwise.cpp`
4. Python wrapper in `_elementwise_funcs.py`
5. Export in `__init__.py`
6. Tests in `../tests/elementwise/`
