---
applyTo:
  - "dpctl/**/*.pyx"
  - "dpctl/**/*.pxd"
  - "dpctl/**/*.pxi"
---

# Cython Instructions

See [dpctl/AGENTS.md](/dpctl/AGENTS.md) for conventions and patterns.

## Quick Reference

### Required Directives (after license header)
```cython
# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
```

### Key Rules
- `cimport` for C-level declarations, `import` for Python
- Store C refs as `_*_ref`, clean up in `__dealloc__`
- Use `with nogil:` for blocking C operations
- Check NULL before using C API returns
