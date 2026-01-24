# AGENTS.md - AI Agent Guide for DPCTL

## Overview

**DPCTL** (Data Parallel Control) is a Python SYCL binding library for heterogeneous computing. It provides Python wrappers for SYCL runtime objects and implements the Python Array API standard for tensor operations.

- **License:** Apache 2.0 (see `LICENSE`)
- **Copyright:** Intel Corporation

## Architecture

```
Python API  →  Cython Bindings  →  C API           →  SYCL Runtime
   dpctl/       _sycl_*.pyx       libsyclinterface/

dpctl.tensor  →  pybind11  →  C++ Kernels (libtensor/)  →  SYCL Runtime
```

## Directory Guide

| Directory | AGENTS.md | Purpose |
|-----------|-----------|---------|
| `dpctl/` | [dpctl/AGENTS.md](dpctl/AGENTS.md) | Core SYCL bindings (Device, Queue, Context) |
| `dpctl/tensor/` | [dpctl/tensor/AGENTS.md](dpctl/tensor/AGENTS.md) | Array API tensor operations |
| `dpctl/tensor/libtensor/` | [dpctl/tensor/libtensor/AGENTS.md](dpctl/tensor/libtensor/AGENTS.md) | C++ SYCL kernels |
| `dpctl/memory/` | [dpctl/memory/AGENTS.md](dpctl/memory/AGENTS.md) | USM memory management |
| `dpctl/program/` | [dpctl/program/AGENTS.md](dpctl/program/AGENTS.md) | SYCL kernel compilation |
| `dpctl/utils/` | [dpctl/utils/AGENTS.md](dpctl/utils/AGENTS.md) | Utility functions |
| `dpctl/tests/` | [dpctl/tests/AGENTS.md](dpctl/tests/AGENTS.md) | Test suite |
| `libsyclinterface/` | [libsyclinterface/AGENTS.md](libsyclinterface/AGENTS.md) | C API layer |

## Code Style

Configuration files (do not hardcode versions - check these files):
- **Python/Cython:** `.pre-commit-config.yaml`
- **C/C++:** `.clang-format`
- **Linting:** `.flake8`

## License Header

All source files require Apache 2.0 header with Intel copyright. Reference existing files for exact format.

## Quick Reference

```python
import dpctl
import dpctl.tensor as dpt

q = dpctl.SyclQueue("gpu")                          # Create queue
x = dpt.ones((100, 100), dtype="f4", sycl_queue=q)  # Create array
np_array = dpt.asnumpy(x)                           # Transfer to host
```

## Key Concepts

- **Queue:** Execution context binding device + context
- **USM:** Unified Shared Memory (device/shared/host types)
- **Filter string:** Device selector syntax `"backend:device_type:num"`
- **Array API:** Python standard for array operations (https://data-apis.org/array-api/)
