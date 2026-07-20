# dpctl/program/ - SYCL Kernel Compilation

## Purpose

Compile and manage SYCL kernels from OpenCL C or SPIR-V source.

## Key Files

| File | Purpose |
|------|---------|
| `_program.pyx` | `SyclProgram`, `SyclKernel` extension types |
| `_program.pxd` | Cython declarations |
| `__init__.py` | Public API exports |

## Classes

- **`SyclProgram`** - Compiled SYCL program containing one or more kernels
- **`SyclKernel`** - Individual kernel extracted from a program

## Usage Pattern

```python
from dpctl.program import create_program_from_source

source = """
__kernel void add(__global float* a, __global float* b, __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

program = create_program_from_source(queue, source)
kernel = program.get_sycl_kernel("add")
```

## Notes

- Programs are context-bound
- Follows same Cython patterns as core dpctl (see `../AGENTS.md`)
