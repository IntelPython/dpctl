---
applyTo:
  - "dpctl/**/*.pyx"
  - "dpctl/**/*.pxd"
  - "dpctl/**/*.pxi"
---

# Cython Bindings Instructions

## Context

Cython files in dpctl provide the bridge between Python and the C/C++ SYCL interface. They wrap SYCL runtime objects as Python extension types.

## File Types

| Extension | Purpose |
|-----------|---------|
| `.pyx` | Implementation files (compiled to C++) |
| `.pxd` | Declaration files (like C headers) |
| `.pxi` | Include files (textually included) |

## Required Directives

Every `.pyx` file must start with (after license header):

```cython
# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
```

## Import Conventions

### cimport vs import

```cython
# cimport - for C-level declarations (compile-time)
from cpython cimport pycapsule
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from ._backend cimport DPCTLSyclDeviceRef, DPCTLDevice_Create

# import - for Python-level usage (runtime)
import numpy as np
from . import _device_selection
```

### Import Order
1. Standard library cimports
2. Third-party cimports
3. Local cimports (with `# noqa: E211` if needed)
4. Blank line
5. Standard library imports
6. Third-party imports
7. Local imports

## Extension Type Pattern

```cython
cdef class SyclDevice:
    """
    Python wrapper for sycl::device.

    Docstring describing the class.
    """
    # C-level attribute (not accessible from Python)
    cdef DPCTLSyclDeviceRef _device_ref

    def __cinit__(self, filter_string=None):
        """
        Called before __init__, handles C memory allocation.
        Must not raise Python exceptions that leave C state invalid.
        """
        if filter_string is not None:
            self._device_ref = DPCTLDevice_CreateFromSelector(...)
        else:
            self._device_ref = NULL

    def __dealloc__(self):
        """
        Called during garbage collection. Clean up C resources.
        """
        if self._device_ref is not NULL:
            DPCTLDevice_Delete(self._device_ref)

    cdef DPCTLSyclDeviceRef get_device_ref(self):
        """
        Internal method for C-level access.
        Not visible from Python.
        """
        return self._device_ref

    @property
    def name(self):
        """Python property wrapping C getter."""
        cdef const char *name_ptr = DPCTLDevice_GetName(self._device_ref)
        if name_ptr is NULL:
            raise RuntimeError("Failed to get device name")
        try:
            return name_ptr.decode("utf-8")
        finally:
            DPCTLCString_Delete(name_ptr)
```

## Memory Management

### Rule: Always Clean Up in __dealloc__

```cython
def __dealloc__(self):
    # Check for NULL before deleting
    if self._queue_ref is not NULL:
        DPCTLQueue_Delete(self._queue_ref)
```

### Ownership Annotations

Match C API annotations:
- `__dpctl_give` - Caller receives ownership, must delete
- `__dpctl_take` - Function takes ownership, don't use after
- `__dpctl_keep` - Function only observes, doesn't take ownership

```cython
cdef void example():
    # Receives ownership (__dpctl_give) - must delete
    cdef DPCTLSyclEventRef event = DPCTLQueue_Submit(...)

    # ... use event ...

    # Clean up owned resource
    DPCTLEvent_Delete(event)
```

### GIL Management

Release GIL for blocking C operations:

```cython
cdef void copy_with_wait(DPCTLSyclEventRef event):
    with nogil:
        DPCTLEvent_Wait(event)

# Or for longer sections
cdef void long_operation() nogil:
    # Entire function runs without GIL
    # Cannot call Python objects here
    pass
```

## Error Handling

### NULL Checks

```cython
cdef DPCTLSyclDeviceRef dref = DPCTLDevice_Create(...)
if dref is NULL:
    raise SyclDeviceCreationError("Failed to create device")
```

### Exception Safety

```cython
def create_something(self):
    cdef SomeRef ref = SomeFunction()
    if ref is NULL:
        raise SomeError("Creation failed")
    try:
        # Operations that might raise
        result = self._process(ref)
    except:
        # Clean up on exception
        SomeDelete(ref)
        raise
    return result
```

## Backend Declarations (.pxd)

```cython
# _backend.pxd
cdef extern from "syclinterface/dpctl_sycl_device_interface.h":
    ctypedef void* DPCTLSyclDeviceRef

    DPCTLSyclDeviceRef DPCTLDevice_Create(
        DPCTLDeviceSelectorRef DSRef
    ) nogil

    void DPCTLDevice_Delete(
        DPCTLSyclDeviceRef DRef
    ) nogil

    const char* DPCTLDevice_GetName(
        DPCTLSyclDeviceRef DRef
    ) nogil
```

## Include Files (.pxi)

Used for shared code snippets:

```cython
# In main .pyx file
include "_sycl_usm_array_interface_utils.pxi"
```

## Common Patterns

### Property with C String Return

```cython
@property
def driver_version(self):
    cdef const char* ver = DPCTLDevice_GetDriverVersion(self._device_ref)
    if ver is NULL:
        return ""
    try:
        return ver.decode("utf-8")
    finally:
        DPCTLCString_Delete(ver)
```

### Capsule for Opaque Pointers

```cython
def get_capsule(self):
    """Return PyCapsule containing C pointer."""
    if self._device_ref is NULL:
        return None
    return pycapsule.PyCapsule_New(
        <void*>self._device_ref,
        "SyclDeviceRef",
        NULL  # No destructor - we manage lifetime
    )
```

### Type Checking

```cython
def some_method(self, other):
    if not isinstance(other, SyclDevice):
        raise TypeError(
            f"Expected SyclDevice, got {type(other).__name__}"
        )
    cdef SyclDevice other_dev = <SyclDevice>other
    # Now can access other_dev._device_ref
```
