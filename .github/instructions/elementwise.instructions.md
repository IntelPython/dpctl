---
applyTo:
  - "dpctl/tensor/libtensor/**/elementwise_functions/**"
  - "dpctl/tensor/_elementwise_*.py"
  - "dpctl/tests/elementwise/**"
---

# Elementwise Operations Instructions

Full stack: C++ kernel → pybind11 → Python wrapper → tests

## References

- **C++ kernels:** [dpctl/tensor/libtensor/AGENTS.md](/dpctl/tensor/libtensor/AGENTS.md)
- **Python wrappers:** [dpctl/tensor/AGENTS.md](/dpctl/tensor/AGENTS.md)
- **Tests:** [dpctl/tests/AGENTS.md](/dpctl/tests/AGENTS.md)

## Adding New Operation Checklist

1. `libtensor/include/kernels/elementwise_functions/op.hpp` - functor
2. `libtensor/source/elementwise_functions/op.cpp` - dispatch tables
3. Register in `tensor_elementwise.cpp`
4. `_elementwise_funcs.py` - Python wrapper
5. Export in `__init__.py`
6. `tests/elementwise/test_op.py` - full coverage
