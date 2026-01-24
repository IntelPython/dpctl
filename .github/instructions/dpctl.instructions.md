---
applyTo:
  - "**/*.py"
  - "**/*.pyx"
  - "**/*.pxd"
  - "**/*.pxi"
  - "**/*.cpp"
  - "**/*.hpp"
  - "**/*.h"
  - "**/*.c"
---

# DPCTL General Instructions

## Project Context

DPCTL (Data Parallel Control) is a Python SYCL binding library for heterogeneous computing. It provides:
- Python wrappers for SYCL runtime objects (devices, queues, contexts)
- Array API-compliant tensor operations (`dpctl.tensor`)
- USM memory management (`dpctl.memory`)
- Kernel compilation (`dpctl.program`)

## License Header Requirement

**All source files must include the Apache 2.0 license header.**

### Python/Cython Files
```python
#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### C/C++ Files
```cpp
//===-- filename.hpp - Brief description              -*-C++-*-===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...
//===----------------------------------------------------------------------===//
```

## Code Style

### Python
- **Formatter:** Black (line length 80)
- **Import sorting:** isort
- **Linting:** flake8
- **Max line length:** 80 characters

### C/C++
- **Formatter:** clang-format (LLVM-based)
- **Indent:** 4 spaces
- **Brace style:** Break before functions, classes, namespaces

## Naming Conventions

### Python/Cython
- Classes: `PascalCase` (e.g., `SyclDevice`, `SyclQueue`)
- Functions/methods: `snake_case` (e.g., `get_device_info`)
- Private: prefix with `_` (e.g., `_validate_queue`)
- Constants: `UPPER_SNAKE_CASE`

### C API (libsyclinterface)
- Functions: `DPCTL<Class>_<Method>` (e.g., `DPCTLDevice_Create`)
- Types: `DPCTL<Class>Ref` (e.g., `DPCTLSyclDeviceRef`)
- Macros: `DPCTL_` prefix

### C++ Kernels
- Functors: `<Operation>Functor` (e.g., `AddFunctor`)
- Kernel classes: `<operation>_krn` suffix
- Namespaces: `dpctl::tensor::kernels`

## Error Handling

### Python Exceptions
- `SyclDeviceCreationError` - Device creation failed
- `SyclQueueCreationError` - Queue creation failed
- `ExecutionPlacementError` - Queue mismatch between arrays
- `TypeError`, `ValueError` - Standard Python errors for invalid input

### C API Error Handling
- Return `NULL` for failed object creation
- Use `error_handler_callback` for async errors
- Check return values before using

## Device Compatibility

**Warning:** Not all SYCL devices support all data types.

- **fp64 (double):** Check `device.has_aspect_fp64`
- **fp16 (half):** Check `device.has_aspect_fp16`

Always use `skip_if_dtype_not_supported()` in tests.

## Common Patterns

### Queue Validation
```python
def some_operation(x, y):
    exec_q = dpctl.utils.get_execution_queue([x.sycl_queue, y.sycl_queue])
    if exec_q is None:
        raise ExecutionPlacementError("...")
```

### Resource Cleanup
```python
cdef class SyclDevice:
    cdef DPCTLSyclDeviceRef _device_ref

    def __dealloc__(self):
        DPCTLDevice_Delete(self._device_ref)
```

### GIL Release for Blocking Operations
```cython
with nogil:
    DPCTLEvent_Wait(event_ref)
```
