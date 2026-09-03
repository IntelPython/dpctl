# dpctl/memory/ - USM Memory Management

## Purpose

Python classes for SYCL Unified Shared Memory (USM) allocation.

## USM Types

| Class | USM Type | Description |
|-------|----------|-------------|
| `MemoryUSMDevice` | Device | Device-only, fastest access |
| `MemoryUSMShared` | Shared | Host and device accessible |
| `MemoryUSMHost` | Host | Host memory, device accessible |

## __sycl_usm_array_interface__

All memory classes implement this protocol:
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

## Memory Lifetime Rules

1. **Queue-bound:** Memory tied to specific queue/context
2. **Outlive operations:** Keep memory alive until operations complete
3. **Views extend lifetime:** Views keep base memory alive

## Key Files

| File | Purpose |
|------|---------|
| `_memory.pyx` | Memory class implementations |
| `_memory.pxd` | Cython declarations |
| `__init__.py` | Public API exports |
