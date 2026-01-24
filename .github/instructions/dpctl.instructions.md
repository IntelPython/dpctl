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

See [/AGENTS.md](/AGENTS.md) for project overview and architecture.

## Key References

- **Code style:** See `.pre-commit-config.yaml` for tool versions, `.clang-format` for C++ style
- **License:** Apache 2.0 with Intel copyright - see existing files for header format
- **Directory guides:** Each major directory has its own `AGENTS.md`

## Critical Rules

1. **Device compatibility:** Not all devices support fp64/fp16 - check capabilities
2. **Queue consistency:** All arrays in an operation must share compatible queues
3. **Resource cleanup:** Always clean up C resources in `__dealloc__`
4. **NULL checks:** Always check C API returns before use
