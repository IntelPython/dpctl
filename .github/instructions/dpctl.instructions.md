---
applyTo:
  - "**/*.py"
  - "**/*.pyx"
  - "**/*.pxd"
  - "**/*.cpp"
  - "**/*.hpp"
  - "**/*.h"
---

# DPCTL General Instructions

See `AGENTS.md` at repository root for project overview and architecture.
Each major directory has its own `AGENTS.md` with specific conventions.

## Key References

- **Code style:** `.pre-commit-config.yaml`, `.clang-format`, `.flake8`
- **License:** Apache 2.0 with Intel copyright - match existing file headers

## Critical Rules

1. **Device compatibility:** Not all devices support fp64/fp16 - never assume availability
2. **Queue consistency:** Arrays in same operation must share compatible queues
3. **Resource cleanup:** Clean up C resources in `__dealloc__` with NULL check
4. **NULL checks:** Always check C API returns before use
