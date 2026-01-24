---
applyTo:
  - "dpctl/tensor/libtensor/**/*.hpp"
  - "dpctl/tensor/libtensor/**/*.cpp"
---

# C++ SYCL Kernel Instructions

See [dpctl/tensor/libtensor/AGENTS.md](/dpctl/tensor/libtensor/AGENTS.md) for patterns and structure.

## Quick Reference

### Type dispatch
See `include/utils/type_dispatch_building.hpp` for the 14 supported types.

### Key Rules
- Kernel class names must be unique
- Use `if constexpr` for compile-time type branching
- Complex types don't support vectorization
- Return `nullptr` from factory for unsupported types
