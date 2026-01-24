---
applyTo:
  - "dpctl/memory/**"
  - "dpctl/memory/*.py"
  - "dpctl/memory/*.pyx"
  - "dpctl/memory/*.pxd"
  - "**/usm*.py"
  - "**/test_sycl_usm*.py"
---

# USM Memory Instructions

## Context

The `dpctl.memory` module provides Python classes for SYCL Unified Shared Memory (USM) allocation and management. USM allows allocating memory that can be accessed by both host and device.

## USM Types

| Type | Class | Description |
|------|-------|-------------|
| Device | `MemoryUSMDevice` | Memory on device only, fastest device access |
| Shared | `MemoryUSMShared` | Accessible from host and device, automatic migration |
| Host | `MemoryUSMHost` | Host memory accessible from device |

### When to Use Each Type

- **Device**: Best performance for device-only data
- **Shared**: Convenient for data accessed by both host and device
- **Host**: When data primarily lives on host but device needs access

## Memory Classes

### MemoryUSMDevice

```python
import dpctl
from dpctl.memory import MemoryUSMDevice

q = dpctl.SyclQueue()

# Allocate 1024 bytes of device memory
mem = MemoryUSMDevice(1024, queue=q)

# Properties
mem.nbytes          # Size in bytes
mem.sycl_queue      # Associated queue
mem.sycl_device     # Associated device
mem.sycl_context    # Associated context

# Copy from host
import numpy as np
host_data = np.array([1, 2, 3, 4], dtype=np.float32)
mem.copy_from_host(host_data.view(np.uint8))

# Copy to host
result = np.empty(4, dtype=np.float32)
mem.copy_to_host(result.view(np.uint8))
```

### MemoryUSMShared

```python
from dpctl.memory import MemoryUSMShared

# Shared memory - accessible from both host and device
mem = MemoryUSMShared(1024, queue=q)

# Can be used similarly to MemoryUSMDevice
```

### MemoryUSMHost

```python
from dpctl.memory import MemoryUSMHost

# Host memory with device accessibility
mem = MemoryUSMHost(1024, queue=q)
```

## __sycl_usm_array_interface__

All memory classes implement the `__sycl_usm_array_interface__` protocol for interoperability:

```python
# The interface is a dictionary
interface = mem.__sycl_usm_array_interface__

# Structure:
{
    "data": (pointer, readonly_flag),  # Tuple of (int, bool)
    "shape": (nbytes,),                 # Tuple of sizes
    "strides": None,                    # Strides (None for contiguous)
    "typestr": "|u1",                   # Data type string (bytes)
    "version": 1,                       # Interface version
    "syclobj": queue                    # Associated SYCL queue
}
```

### Using the Interface

```python
def accepts_usm_memory(obj):
    """Accept any object with __sycl_usm_array_interface__."""
    if hasattr(obj, "__sycl_usm_array_interface__"):
        iface = obj.__sycl_usm_array_interface__
        ptr, readonly = iface["data"]
        nbytes = iface["shape"][0]
        queue = iface["syclobj"]
        return ptr, nbytes, queue
    raise TypeError("Object does not support __sycl_usm_array_interface__")
```

## Memory Lifetime Rules

### Rule 1: Memory is Queue-Bound

Memory allocations are associated with a specific queue/context:

```python
q1 = dpctl.SyclQueue("gpu:0")
q2 = dpctl.SyclQueue("gpu:1")

mem1 = MemoryUSMDevice(1024, queue=q1)

# Cannot directly use mem1 with q2 if different context
# Must copy through host or peer-to-peer if supported
```

### Rule 2: Memory Outlives Operations

Ensure memory remains valid until all operations complete:

```python
# BAD - memory might be freed before kernel finishes
def bad_example():
    mem = MemoryUSMDevice(1024, queue=q)
    submit_kernel(mem)  # Async operation
    # mem goes out of scope - danger!

# GOOD - wait for completion
def good_example():
    mem = MemoryUSMDevice(1024, queue=q)
    event = submit_kernel(mem)
    event.wait()  # Ensure completion before mem freed
```

### Rule 3: View Keeps Base Alive

Creating views extends lifetime:

```python
base = MemoryUSMDevice(1024, queue=q)
view = base[100:200]  # View into base

# base remains valid as long as view exists
del base  # view still holds reference
```

## Cython Memory Management

### Allocation Pattern

```cython
cdef class MemoryUSMDevice:
    cdef DPCTLSyclUSMRef _memory_ref
    cdef size_t _nbytes
    cdef SyclQueue _queue

    def __cinit__(self, size_t nbytes, SyclQueue queue):
        cdef DPCTLSyclQueueRef qref = queue.get_queue_ref()

        self._memory_ref = DPCTLmalloc_device(nbytes, qref)
        if self._memory_ref is NULL:
            raise MemoryError(f"Failed to allocate {nbytes} bytes")

        self._nbytes = nbytes
        self._queue = queue

    def __dealloc__(self):
        if self._memory_ref is not NULL:
            DPCTLfree_with_queue(
                self._memory_ref,
                self._queue.get_queue_ref()
            )
```

### Copy Operations

```cython
def copy_from_host(self, host_buffer):
    """Copy data from host to device memory."""
    cdef unsigned char[::1] buf = host_buffer

    if <size_t>buf.shape[0] > self._nbytes:
        raise ValueError("Source buffer larger than allocation")

    cdef DPCTLSyclEventRef event = DPCTLQueue_Memcpy(
        self._queue.get_queue_ref(),
        self._memory_ref,
        <void*>&buf[0],
        buf.shape[0]
    )

    with nogil:
        DPCTLEvent_Wait(event)
    DPCTLEvent_Delete(event)
```

## Testing Memory Operations

```python
import pytest
import numpy as np
from dpctl.memory import MemoryUSMDevice, MemoryUSMShared, MemoryUSMHost
from dpctl.tests.helper import get_queue_or_skip


class TestUSMMemory:
    @pytest.mark.parametrize("mem_class", [
        MemoryUSMDevice, MemoryUSMShared, MemoryUSMHost
    ])
    def test_allocation(self, mem_class):
        q = get_queue_or_skip()
        mem = mem_class(1024, queue=q)
        assert mem.nbytes == 1024
        assert mem.sycl_queue == q

    @pytest.mark.parametrize("mem_class", [
        MemoryUSMDevice, MemoryUSMShared, MemoryUSMHost
    ])
    def test_copy_roundtrip(self, mem_class):
        q = get_queue_or_skip()

        src = np.array([1, 2, 3, 4], dtype=np.float32)
        mem = mem_class(src.nbytes, queue=q)

        mem.copy_from_host(src.view(np.uint8))

        dst = np.empty_like(src)
        mem.copy_to_host(dst.view(np.uint8))

        assert np.array_equal(src, dst)

    def test_interface(self):
        q = get_queue_or_skip()
        mem = MemoryUSMDevice(1024, queue=q)

        iface = mem.__sycl_usm_array_interface__
        assert "data" in iface
        assert "shape" in iface
        assert iface["shape"] == (1024,)
        assert iface["syclobj"] == q
```

## Integration with usm_ndarray

`dpctl.tensor.usm_ndarray` uses USM memory internally:

```python
import dpctl.tensor as dpt

# Creates usm_ndarray with underlying USM allocation
x = dpt.ones((100, 100), dtype=dpt.float32, usm_type="device")

# Access memory properties
x.usm_type   # "device", "shared", or "host"
x.sycl_queue # Associated queue

# Get raw memory (if needed)
# Note: usm_ndarray manages its own memory
```

## Best Practices

1. **Use usm_ndarray for arrays** - Prefer `dpctl.tensor` over raw memory
2. **Match queues** - Ensure arrays/memory share compatible queues
3. **Wait before access** - Wait for async operations before host access
4. **Use appropriate USM type** - Device for performance, shared for convenience
5. **Check allocation success** - Handle MemoryError for large allocations
