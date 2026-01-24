---
applyTo:
  - "dpctl/tensor/libtensor/**/*.hpp"
  - "dpctl/tensor/libtensor/**/*.cpp"
---

# C++ SYCL Kernel Instructions

See `dpctl/tensor/libtensor/AGENTS.md` for patterns and directory structure.

## Key Rules
- Kernel class names must be globally unique
- Use `if constexpr` for compile-time type branching
- Complex types don't support vectorization
- Return `nullptr` from factory for unsupported types
- Check `include/kernels/elementwise_functions/common.hpp` for base patterns
