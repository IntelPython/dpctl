# dpctl/tensor/ - Array API Tensor Operations

## Purpose

Python Array API-compliant tensor operations using `usm_ndarray` on SYCL devices.

## Key Files

| File | Purpose |
|------|---------|
| `_usmarray.pyx` | `usm_ndarray` extension type |
| `_elementwise_funcs.py` | Elementwise operation wrappers |
| `_elementwise_common.py` | `UnaryElementwiseFunc`, `BinaryElementwiseFunc` base classes |
| `_reduction.py` | Reduction operations (sum, prod, mean, etc.) |
| `_manipulation_functions.py` | reshape, concat, stack, split, etc. |
| `_ctors.py` | Array constructors (empty, zeros, ones, arange, etc.) |
| `_type_utils.py` | Type promotion and validation |
| `_sorting.py` | sort, argsort |
| `_searchsorted.py` | searchsorted, digitize |
| `_dlpack.pyx` | DLPack interoperability |
| `_copy_utils.py` | Copy and type casting |

See [libtensor/AGENTS.md](libtensor/AGENTS.md) for C++ kernel implementation.

## Elementwise Wrapper Pattern

```python
from ._elementwise_common import UnaryElementwiseFunc
import dpctl.tensor._tensor_impl as ti

abs = UnaryElementwiseFunc(
    "abs",                # Operation name
    ti._abs_result_type,  # Type inference function
    ti._abs,              # Kernel implementation (from pybind11)
    _abs_docstring_
)
```

## Queue Validation (required for all operations)

```python
exec_q = dpctl.utils.get_execution_queue([x.sycl_queue, y.sycl_queue])
if exec_q is None:
    raise ExecutionPlacementError("Arrays on incompatible queues")
```

## Adding New Operations

1. C++ kernel header: `libtensor/include/kernels/<category>/op.hpp`
2. C++ source: `libtensor/source/<category>/op.cpp`
3. Register in appropriate `tensor_*.cpp` entry point
4. Python wrapper in appropriate `_*.py` module
5. Export in `__init__.py`
6. Tests in `../tests/` with full dtype/usm coverage
