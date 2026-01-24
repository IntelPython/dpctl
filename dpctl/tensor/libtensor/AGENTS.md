# dpctl/tensor/libtensor/ - C++ SYCL Kernels

## Purpose

C++ SYCL kernel implementations for tensor operations, exposed to Python via pybind11.

## Directory Structure

```
libtensor/
├── include/
│   ├── kernels/
│   │   ├── elementwise_functions/  # Unary/binary operations
│   │   │   ├── common.hpp          # Base iteration patterns
│   │   │   ├── common_inplace.hpp  # In-place patterns
│   │   │   └── <op>.hpp            # Individual operations
│   │   ├── linalg_functions/       # GEMM, dot product
│   │   ├── sorting/                # Sort, searchsorted
│   │   ├── reductions.hpp
│   │   └── accumulators.hpp
│   └── utils/
│       ├── type_dispatch.hpp           # Runtime type lookup
│       ├── type_dispatch_building.hpp  # Dispatch table generation
│       ├── offset_utils.hpp            # Stride/offset calculations
│       └── sycl_alloc_utils.hpp        # USM allocation helpers
├── source/
│   ├── elementwise_functions/
│   ├── reductions/
│   ├── sorting/
│   ├── linalg_functions/
│   └── tensor_*.cpp                    # pybind11 module definitions
└── tests/                              # C++ unit tests
```

## Type Dispatch

14 supported types - see `include/utils/type_dispatch_building.hpp` for `typenum_t` enum.

## Kernel Functor Pattern

```cpp
template <typename argT, typename resT>
struct MyOpFunctor {
    // Vectorization hints (check common.hpp for usage)
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    // Scalar operation (required)
    resT operator()(const argT &x) const {
        return /* computation */;
    }

    // Vectorized operation (optional, when supports_vec is true)
    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(
        const sycl::vec<argT, vec_sz> &x) const {
        return /* vectorized computation */;
    }
};
```

## Factory Pattern

```cpp
template <typename fnT, typename argT, typename resT>
struct MyOpContigFactory {
    fnT get() {
        if constexpr (/* unsupported type combination */) {
            return nullptr;  // Signals unsupported
        } else {
            return my_op_contig_impl<argT, resT>;
        }
    }
};
```

## Key Rules

- Kernel class names must be globally unique
- Use `if constexpr` for compile-time type branching
- Complex types typically don't support vectorization
- Return `nullptr` from factory for unsupported type combinations
- Check `common.hpp` and existing operations for patterns
