---
applyTo:
  - "dpctl/**/*.pyx"
  - "dpctl/**/*.pxd"
  - "dpctl/**/*.pxi"
---

# Cython Instructions

See `dpctl/AGENTS.md` for full conventions.

## Required Directives (after license)
```cython
# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
```

## Key Rules
- `cimport` for C-level, `import` for Python-level
- Store C refs as `_*_ref`, clean up in `__dealloc__` with NULL check
- Use `with nogil:` for blocking C operations
