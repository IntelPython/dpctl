---
applyTo:
  - "dpctl/tensor/*.py"
  - "dpctl/tensor/**/*.py"
---

# Tensor Python Code Instructions

## Context

The `dpctl.tensor` module provides Array API-compliant tensor operations. It wraps C++ SYCL kernels with Python interfaces that follow the Python Array API standard.

## Key Files

| File | Purpose |
|------|---------|
| `_usmarray.pyx` | `usm_ndarray` extension type |
| `_elementwise_funcs.py` | Elementwise operation wrappers |
| `_elementwise_common.py` | Base classes for elementwise ops |
| `_reduction.py` | Reduction operations |
| `_manipulation_functions.py` | reshape, stack, concat, etc. |
| `_ctors.py` | Array constructors |
| `_copy_utils.py` | Copy and cast utilities |
| `_type_utils.py` | Type promotion logic |
| `_device.py` | Device object for Array API |

## Array API Compliance

DPCTL tensor aims for Array API compatibility. Key requirements:

1. **Immutable dtype and device** - Arrays don't change type or device implicitly
2. **Explicit data movement** - Use `asarray(x, device=...)` for device transfers
3. **Type promotion rules** - Follow Array API promotion table
4. **No implicit broadcasting** - Some operations require explicit broadcast

## Queue Validation Pattern

All operations must validate that input arrays share a compatible execution context:

```python
def some_operation(x, y):
    """Example binary operation."""
    if not isinstance(x, usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(x)}")
    if not isinstance(y, usm_ndarray):
        raise TypeError(f"Expected usm_ndarray, got {type(y)}")

    # Get common queue (returns None if incompatible)
    exec_q = dpctl.utils.get_execution_queue(
        [x.sycl_queue, y.sycl_queue]
    )
    if exec_q is None:
        raise ExecutionPlacementError(
            "Input arrays must share the same SYCL queue or be on "
            "compatible queues."
        )

    # Proceed with operation
    return _impl(x, y, sycl_queue=exec_q)
```

## Order Manager

Use `OrderManager` for handling memory order (C/F contiguous):

```python
from dpctl.utils import OrderManager

def operation_with_order(x, order="K"):
    """
    Args:
        order: Memory order for output.
            - "C": C-contiguous (row-major)
            - "F": Fortran-contiguous (column-major)
            - "A": F if input is F-contiguous, else C
            - "K": Keep input order
    """
    om = OrderManager(order)

    # Determine output order based on input
    out_order = om.get_order(x)

    # Create output array with determined order
    result = empty(x.shape, dtype=x.dtype, order=out_order,
                   sycl_queue=x.sycl_queue)
    return result
```

## Elementwise Operation Wrapper

```python
# In _elementwise_funcs.py

from ._elementwise_common import (
    UnaryElementwiseFunc,
    BinaryElementwiseFunc,
)
import dpctl.tensor._tensor_impl as ti

_abs_docstring_ = """
abs(x, /, out=None, order="K")

Computes the absolute value element-wise.

Args:
    x (usm_ndarray): Input array.
    out (usm_ndarray, optional): Output array.
    order ({"K", "C", "F", "A"}): Memory order of output.

Returns:
    usm_ndarray: Absolute values.
"""

abs = UnaryElementwiseFunc(
    "abs",                    # Operation name
    ti._abs_result_type,      # Result type inference function
    ti._abs,                  # Kernel implementation
    _abs_docstring_           # Docstring
)
```

## Docstring Format

Follow NumPy-style docstrings:

```python
def reshape(x, /, shape, *, order="C", copy=None):
    """
    reshape(x, shape, order="C", copy=None)

    Gives a new shape to an array without changing its data.

    Args:
        x (usm_ndarray):
            Input array to reshape.
        shape (tuple of ints):
            New shape. One dimension may be -1, which is inferred.
        order ({"C", "F"}, optional):
            Read elements in this order. Default: "C".
        copy (bool, optional):
            If True, always copy. If False, never copy (raise if needed).
            If None, copy only when necessary. Default: None.

    Returns:
        usm_ndarray:
            Reshaped array. May be a view or a copy.

    Raises:
        ValueError:
            If the new shape is incompatible with the original shape.
        TypeError:
            If input is not a usm_ndarray.

    Examples:
        >>> import dpctl.tensor as dpt
        >>> x = dpt.arange(6)
        >>> dpt.reshape(x, (2, 3))
        usm_ndarray([[0, 1, 2],
                     [3, 4, 5]])
    """
```

## Type Utilities

```python
from ._type_utils import (
    _to_device_supported_dtype,
    _resolve_one_strong_one_weak_types,
    _resolve_one_strong_two_weak_types,
)

def some_operation(x, scalar):
    # Resolve type when mixing array and scalar
    res_dtype = _resolve_one_strong_one_weak_types(
        x.dtype,
        type(scalar),
        x.sycl_device
    )

    # Ensure dtype is supported on device
    res_dtype = _to_device_supported_dtype(res_dtype, x.sycl_device)
```

## Common Imports

```python
import dpctl
import dpctl.tensor as dpt
from dpctl.tensor import usm_ndarray
from dpctl.tensor._type_utils import _to_device_supported_dtype
from dpctl.utils import ExecutionPlacementError
import dpctl.tensor._tensor_impl as ti  # C++ bindings

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
```

## Error Messages

Provide clear, actionable error messages:

```python
# Good
raise TypeError(
    f"Expected usm_ndarray for argument 'x', got {type(x).__name__}"
)

# Good
raise ValueError(
    f"Cannot reshape array of size {x.size} into shape {shape}"
)

# Good
raise ExecutionPlacementError(
    "Input arrays are allocated on incompatible devices. "
    "Use dpctl.tensor.asarray(x, device=target_device) to copy."
)
```

## Output Array Pattern

Many operations accept an optional `out` parameter:

```python
def operation(x, /, out=None, order="K"):
    # Validate input
    if not isinstance(x, usm_ndarray):
        raise TypeError(...)

    # Determine output properties
    res_dtype = _compute_result_type(x.dtype)
    res_shape = x.shape

    if out is not None:
        # Validate output array
        if not isinstance(out, usm_ndarray):
            raise TypeError("out must be usm_ndarray")
        if out.shape != res_shape:
            raise ValueError(f"out shape {out.shape} != {res_shape}")
        if out.dtype != res_dtype:
            raise ValueError(f"out dtype {out.dtype} != {res_dtype}")
        if out.sycl_queue != x.sycl_queue:
            raise ValueError("out must be on same queue as input")
    else:
        # Create output array
        out = empty(res_shape, dtype=res_dtype, order=order,
                    sycl_queue=x.sycl_queue)

    # Call implementation
    _impl(x, out)
    return out
```

## Best Practices

1. **Validate inputs early** - Check types and shapes before computation
2. **Use get_execution_queue** - Ensure queue compatibility
3. **Support order parameter** - Allow C/F/A/K memory orders
4. **Return same queue** - Output arrays on same queue as inputs
5. **Follow Array API** - Match standard function signatures
6. **Document thoroughly** - Include types, shapes, exceptions in docstrings
