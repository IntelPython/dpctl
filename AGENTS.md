# AGENTS.md - AI Agent Guide for DPCTL

## Overview

**DPCTL** (Data Parallel Control) is a Python library providing bindings for SYCL, enabling heterogeneous computing on CPUs, GPUs, and accelerators. It implements the Python Array API standard for tensor operations on SYCL devices.

- **License:** Apache 2.0
- **Copyright:** Intel Corporation
- **Primary Language:** Python with Cython bindings and C++ SYCL kernels

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python API Layer                            │
│  dpctl, dpctl.tensor, dpctl.memory, dpctl.program, dpctl.utils  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cython Bindings                              │
│  _sycl_device.pyx, _sycl_queue.pyx, _usmarray.pyx, etc.        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      C API Layer                                │
│               libsyclinterface (dpctl_*.h)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   C++ Kernel Layer                              │
│         dpctl/tensor/libtensor (SYCL kernels)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYCL Runtime                                 │
│              (Intel oneAPI, OpenSYCL, etc.)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Module Relationships

- **dpctl**: Core SYCL object wrappers (Device, Queue, Context, Event, Platform)
- **dpctl.tensor**: Array API-compliant tensor operations using `usm_ndarray`
- **dpctl.memory**: USM (Unified Shared Memory) allocation classes
- **dpctl.program**: SYCL kernel compilation and program management
- **dpctl.utils**: Utility functions for device queries and order management

---

## Module Reference

### dpctl/ (Core SYCL Bindings)

**Purpose:** Python wrappers for fundamental SYCL runtime objects.

**Key Files:**
| File | Purpose |
|------|---------|
| `_sycl_device.pyx` | `SyclDevice` - wraps `sycl::device` |
| `_sycl_queue.pyx` | `SyclQueue` - wraps `sycl::queue` |
| `_sycl_context.pyx` | `SyclContext` - wraps `sycl::context` |
| `_sycl_event.pyx` | `SyclEvent` - wraps `sycl::event` |
| `_sycl_platform.pyx` | `SyclPlatform` - wraps `sycl::platform` |
| `_backend.pxd` | C API declarations for Cython |
| `enum_types.py` | Python enums for SYCL types |

**Conventions:**
- Extension types use `cdef class` with C reference as `_*_ref` attribute
- All C resources cleaned up in `__dealloc__`
- Use `nogil` when calling blocking C operations
- Exceptions: `SyclDeviceCreationError`, `SyclQueueCreationError`, etc.

### dpctl/tensor/ (Array API)

**Purpose:** Python Array API-compliant tensor operations on SYCL devices.

**Key Files:**
| File | Purpose |
|------|---------|
| `_usmarray.pyx` | `usm_ndarray` extension type |
| `_elementwise_funcs.py` | Elementwise operation wrappers |
| `_elementwise_common.py` | `UnaryElementwiseFunc`, `BinaryElementwiseFunc` |
| `_reduction.py` | Reduction operations (sum, prod, etc.) |
| `_manipulation_functions.py` | reshape, concat, stack, etc. |
| `_ctors.py` | Array constructors (empty, zeros, ones, etc.) |
| `_copy_utils.py` | Copy and type casting utilities |
| `_type_utils.py` | Type promotion and validation |

**Type Dispatch Pattern:**
```python
# In _elementwise_funcs.py
abs = UnaryElementwiseFunc(
    "abs",                    # Operation name
    ti._abs_result_type,      # Type inference function
    ti._abs,                  # Kernel implementation
    _abs_docstring_           # Documentation
)
```

**Queue Validation Pattern:**
```python
def validate_queues(q1, q2):
    """Ensure arrays share execution context"""
    if q1 != q2:
        raise ExecutionPlacementError(...)
```

### dpctl/tensor/libtensor/ (C++ Kernels)

**Purpose:** SYCL kernel implementations for tensor operations.

**Directory Structure:**
```
libtensor/
├── include/
│   ├── kernels/
│   │   ├── elementwise_functions/  # 100+ operations
│   │   │   ├── common.hpp          # Base patterns
│   │   │   ├── add.hpp, sin.hpp, etc.
│   │   ├── linalg_functions/       # GEMM, dot
│   │   ├── sorting/                # Sort algorithms
│   │   └── reductions.hpp
│   └── utils/
│       ├── type_dispatch.hpp
│       ├── type_dispatch_building.hpp
│       └── offset_utils.hpp
└── source/
    ├── elementwise_functions/
    ├── reductions/
    └── tensor_*.cpp                # Entry points
```

**Type Enumeration (14 types):**
```cpp
enum class typenum_t : int {
    BOOL = 0,
    INT8, UINT8, INT16, UINT16, INT32, UINT32,
    INT64, UINT64, HALF, FLOAT, DOUBLE,
    CFLOAT, CDOUBLE,
};
```

**Kernel Functor Pattern:**
```cpp
template <typename argT, typename resT>
struct MyFunctor {
    using supports_sg_loadstore = std::true_type;  // Vectorization hint

    resT operator()(const argT &x) const {
        return /* computation */;
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(
        const sycl::vec<argT, vec_sz> &x) const {
        return /* vectorized computation */;
    }
};
```

### dpctl/memory/ (USM Memory)

**Purpose:** Unified Shared Memory allocation and management.

**Key Classes:**
- `MemoryUSMDevice` - Device-only memory
- `MemoryUSMShared` - Host and device accessible
- `MemoryUSMHost` - Host memory accessible from device

**Interface:**
All classes implement `__sycl_usm_array_interface__`:
```python
{
    "data": (ptr, readonly_flag),
    "shape": (nbytes,),
    "strides": None,
    "typestr": "|u1",
    "version": 1,
    "syclobj": queue
}
```

### dpctl/program/ (Kernel Compilation)

**Purpose:** Compile and manage SYCL kernels from source.

**Key Classes:**
- `SyclProgram` - Compiled SYCL program
- `SyclKernel` - Individual kernel extracted from program

### libsyclinterface/ (C API)

**Purpose:** C wrapper around SYCL C++ runtime for language interoperability.

**Naming Convention:** `DPCTL<Class>_<Method>`
```c
DPCTLDevice_Create()
DPCTLQueue_Submit()
DPCTLContext_Delete()
```

**Memory Ownership Annotations:**
| Annotation | Meaning |
|------------|---------|
| `__dpctl_give` | Caller receives ownership, must free |
| `__dpctl_take` | Function takes ownership, caller must not use after |
| `__dpctl_keep` | Function observes only, does not take ownership |
| `__dpctl_null` | NULL value expected or may be returned |

**Example:**
```c
DPCTL_API
__dpctl_give DPCTLSyclContextRef
DPCTLContext_Create(__dpctl_keep const DPCTLSyclDeviceRef DRef,
                    error_handler_callback *handler,
                    int properties);
```

---

## Code Style

### Python/Cython
- **Formatter:** Black (line length 80)
- **Import Sort:** isort
- **Linting:** flake8, cython-lint
- **String Quotes:** Double quotes for Cython

### C/C++
- **Formatter:** clang-format (LLVM-based style)
- **Indent:** 4 spaces
- **Braces:** Break before functions, classes, namespaces

### License Header (Required in ALL files)

**Python/Cython:**
```python
#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
```

**C/C++:**
```cpp
//===-- filename.hpp - Description                    -*-C++-*-===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...
//===----------------------------------------------------------------------===//
```

---

## Testing Guidelines

### Test Location
- Main tests: `dpctl/tests/`
- Elementwise tests: `dpctl/tests/elementwise/`
- Libtensor tests: `dpctl/tensor/libtensor/tests/`

### Key Fixtures (`conftest.py`)
```python
get_queue_or_skip()           # Create queue or skip test
skip_if_dtype_not_supported() # Skip if device lacks dtype support
```

### Dtype Coverage
All operations should be tested with:
```python
_integral_dtypes = ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8"]
_real_fp_dtypes = ["f2", "f4", "f8"]
_complex_fp_dtypes = ["c8", "c16"]
```

### USM Type Coverage
```python
_usm_types = ["device", "shared", "host"]
```

### Memory Order Coverage
```python
_orders = ["C", "F", "A", "K"]
```

### Test Pattern
```python
@pytest.mark.parametrize("dtype", _all_dtypes)
@pytest.mark.parametrize("usm_type", _usm_types)
def test_operation(dtype, usm_type):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    x = dpt.ones(100, dtype=dtype, usm_type=usm_type, sycl_queue=q)
    result = dpt.operation(x)

    expected = np.operation(dpt.asnumpy(x))
    assert_allclose(dpt.asnumpy(result), expected)
```

---

## Adding New Elementwise Operations

### Checklist

1. **C++ Kernel** (`libtensor/include/kernels/elementwise_functions/`)
   - [ ] Create `operation.hpp` with functor template
   - [ ] Define type support matrix
   - [ ] Implement scalar and vector operations

2. **C++ Source** (`libtensor/source/elementwise_functions/`)
   - [ ] Create `operation.cpp`
   - [ ] Instantiate dispatch tables
   - [ ] Register in `tensor_elementwise.cpp`

3. **Python Wrapper** (`dpctl/tensor/`)
   - [ ] Add to `_elementwise_funcs.py`
   - [ ] Export in `__init__.py`

4. **Tests** (`dpctl/tests/elementwise/`)
   - [ ] Create `test_operation.py`
   - [ ] Cover all dtypes, USM types, orders
   - [ ] Test edge cases (empty, scalar, broadcast)

5. **Documentation** (`docs/`)
   - [ ] Add to API reference
   - [ ] Include in elementwise functions list

---

## Common Pitfalls

### Memory Management
- **Always** clean up C resources in `__dealloc__`
- Use `__dpctl_give`/`__dpctl_take` annotations correctly
- Check for NULL returns from C API functions
- Release GIL (`with nogil:`) during blocking operations

### Type Promotion
- Follow Array API type promotion rules
- Handle mixed real/complex carefully
- Check device support for fp16/fp64 before operations

### Device Capabilities
- Not all devices support fp64 (double precision)
- Not all devices support fp16 (half precision)
- Use `skip_if_dtype_not_supported()` in tests

### Queue Consistency
- All arrays in an operation must share the same queue
- Use `dpctl.utils.get_execution_queue()` to validate

### Broadcasting
- Follow NumPy broadcasting rules
- Handle 0-d arrays (scalars) correctly

---

## Code Review Checklist

### All Code
- [ ] License header present and correct year
- [ ] Code formatted (black/clang-format)
- [ ] No hardcoded device assumptions (fp64, etc.)

### Python/Cython
- [ ] Docstrings for public functions
- [ ] Type hints where applicable
- [ ] Exceptions have descriptive messages
- [ ] Resources cleaned up properly

### C++ Kernels
- [ ] Functor handles all required types
- [ ] Vectorization hints correct (`supports_sg_loadstore`, etc.)
- [ ] No raw `new`/`delete` (use SYCL allocators)
- [ ] Kernel names are unique

### C API
- [ ] Memory ownership annotations on all functions
- [ ] NULL checks for parameters
- [ ] Error handling via callbacks
- [ ] Extern C wrapper for C++ code

### Tests
- [ ] Covers all supported dtypes
- [ ] Covers all USM types
- [ ] Handles device capability limitations
- [ ] Tests edge cases (empty arrays, scalars)
- [ ] Uses appropriate fixtures

---

## Quick Reference

### Filter String Syntax
```
backend:device_type:device_num
```
Examples: `"opencl:gpu:0"`, `"level_zero:gpu"`, `"cpu"`

### Common Imports
```python
import dpctl
import dpctl.tensor as dpt
from dpctl.tensor import usm_ndarray
from dpctl.memory import MemoryUSMShared
```

### Create Queue
```python
q = dpctl.SyclQueue()                    # Default device
q = dpctl.SyclQueue("gpu")               # First GPU
q = dpctl.SyclQueue("level_zero:gpu:0")  # Specific device
```

### Create Array
```python
x = dpt.empty((100, 100), dtype=dpt.float32, sycl_queue=q)
x = dpt.asarray(np_array, device=q)
```

### Transfer Data
```python
np_array = dpt.asnumpy(x)        # Device to host
x = dpt.asarray(np_array, device=q)  # Host to device
```
