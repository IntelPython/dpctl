# dpctl/ - Core SYCL Bindings

## Purpose

Python/Cython wrappers for SYCL runtime objects: Device, Queue, Context, Event, Platform.

## Key Files

| File | Purpose |
|------|---------|
| `_sycl_device.pyx` | `SyclDevice` wrapping `sycl::device` |
| `_sycl_queue.pyx` | `SyclQueue` wrapping `sycl::queue` |
| `_sycl_context.pyx` | `SyclContext` wrapping `sycl::context` |
| `_sycl_event.pyx` | `SyclEvent` wrapping `sycl::event` |
| `_sycl_platform.pyx` | `SyclPlatform` wrapping `sycl::platform` |
| `_backend.pxd` | C API declarations from libsyclinterface |
| `enum_types.py` | Python enums for SYCL types |

## Cython Conventions

### Required Directives
```cython
# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
```

### Extension Type Pattern
```cython
cdef class SyclDevice:
    cdef DPCTLSyclDeviceRef _device_ref  # C reference

    def __dealloc__(self):
        if self._device_ref is not NULL:
            DPCTLDevice_Delete(self._device_ref)

    cdef DPCTLSyclDeviceRef get_device_ref(self):
        return self._device_ref
```

### Key Rules
- Store C references as `_*_ref` attributes
- Always clean up in `__dealloc__`
- Use `with nogil:` for blocking C calls
- Check NULL before using C API returns

### Exceptions
- `SyclDeviceCreationError`
- `SyclQueueCreationError`
- `SyclContextCreationError`

## cimport vs import

```cython
# cimport - C-level declarations (compile-time)
from ._backend cimport DPCTLSyclDeviceRef, DPCTLDevice_Create

# import - Python-level (runtime)
from . import _device_selection
```
