---
applyTo:
  - "dpctl/tensor/libtensor/**/*.hpp"
  - "dpctl/tensor/libtensor/**/*.cpp"
---

# C++ SYCL Kernel Instructions (libtensor)

## Context

The libtensor directory contains C++ SYCL kernel implementations for tensor operations. These kernels are called from Python via pybind11 bindings.

## Directory Structure

```
libtensor/
├── include/
│   ├── kernels/
│   │   ├── elementwise_functions/  # Unary/binary operations
│   │   │   ├── common.hpp          # Base iteration patterns
│   │   │   ├── common_inplace.hpp  # In-place operation patterns
│   │   │   ├── add.hpp, sin.hpp, etc.
│   │   ├── linalg_functions/       # Linear algebra (gemm, dot)
│   │   ├── sorting/                # Sort, argsort, searchsorted
│   │   └── reductions.hpp          # Reduction operations
│   └── utils/
│       ├── type_dispatch.hpp       # Type lookup tables
│       ├── type_dispatch_building.hpp  # Dispatch table generation
│       ├── offset_utils.hpp        # Stride calculations
│       └── sycl_alloc_utils.hpp    # USM allocation helpers
└── source/
    ├── elementwise_functions/      # Implementation files
    ├── reductions/
    ├── sorting/
    └── tensor_*.cpp                # pybind11 entry points
```

## File Header Format

```cpp
//===-- operation.hpp - Brief description             -*-C++-*-===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...
//===----------------------------------------------------------------------===//
///
/// \file
/// Detailed description of the file's purpose and contents.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
// ... other includes
```

## Type Dispatch System

### Type Enumeration

```cpp
// In type_dispatch_building.hpp
enum class typenum_t : int {
    BOOL = 0,
    INT8, UINT8, INT16, UINT16, INT32, UINT32,
    INT64, UINT64, HALF, FLOAT, DOUBLE,
    CFLOAT, CDOUBLE,
};
inline constexpr int num_types = 14;
```

### Dispatch Table Pattern

```cpp
// 2D dispatch table: output_type_fn_ptr_t[dst_type][src_type]
template <typename funcPtrT,
          template <typename fnT, typename D, typename S> typename factory,
          int _num_types>
class DispatchTableBuilder
{
public:
    DispatchTableBuilder() = default;

    void populate_dispatch_table(funcPtrT table[][_num_types]) const
    {
        // Populates table with type-specific function pointers
        // using the factory template
    }
};
```

### Using Dispatch Tables

```cpp
// In operation.cpp
static unary_contig_impl_fn_ptr_t abs_contig_dispatch_table[num_types];

void init_abs_dispatch_tables()
{
    using dpctl::tensor::kernels::abs::AbsContigFactory;

    DispatchTableBuilder<unary_contig_impl_fn_ptr_t,
                         AbsContigFactory,
                         num_types>
        dtb;
    dtb.populate_dispatch_table(abs_contig_dispatch_table);
}
```

## Kernel Functor Pattern

### Unary Operation

```cpp
template <typename argT, typename resT>
struct AbsFunctor
{
    // Type traits for optimization
    using is_constant = std::false_type;
    using supports_sg_loadstore = std::true_type;
    using supports_vec = std::true_type;

    // Scalar operation
    resT operator()(const argT &x) const
    {
        if constexpr (std::is_same_v<argT, bool>) {
            return x;
        }
        else if constexpr (is_complex<argT>::value) {
            return cabs(x);
        }
        else if constexpr (std::is_unsigned_v<argT>) {
            return x;
        }
        else {
            return (x >= argT(0)) ? x : -x;
        }
    }

    // Vectorized operation (when supports_vec is true)
    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT, vec_sz> &x) const
    {
        return sycl::abs(x);
    }
};
```

### Binary Operation

```cpp
template <typename argT1, typename argT2, typename resT>
struct AddFunctor
{
    using supports_sg_loadstore = std::negation<
        std::disjunction<is_complex<argT1>, is_complex<argT2>>>;
    using supports_vec = std::negation<
        std::disjunction<is_complex<argT1>, is_complex<argT2>>>;

    resT operator()(const argT1 &x, const argT2 &y) const
    {
        if constexpr (is_complex<argT1>::value || is_complex<argT2>::value) {
            using rT1 = typename argT1::value_type;
            using rT2 = typename argT2::value_type;
            return exprm_ns::complex<resT>(x) + exprm_ns::complex<resT>(y);
        }
        else {
            return x + y;
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT1, vec_sz> &x,
               const sycl::vec<argT2, vec_sz> &y) const
    {
        return x + y;
    }
};
```

## Kernel Naming

Kernel class names must be unique across the entire codebase:

```cpp
template <typename argT, typename resT>
class abs_contig_kernel;  // Unique name

template <typename argT, typename resT>
class abs_strided_kernel;  // Different name for different pattern
```

## Type Support Matrix

Define which type combinations are supported:

```cpp
template <typename argT, typename resT>
struct AbsOutputType
{
    // Default: not defined (unsupported)
};

template <>
struct AbsOutputType<float, void>
{
    using value_type = float;
};

template <>
struct AbsOutputType<std::complex<float>, void>
{
    using value_type = float;  // abs of complex returns real
};
```

## Factory Pattern

```cpp
template <typename fnT, typename argT, typename resT>
struct AbsContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename AbsOutputType<argT, resT>::value_type,
                          void>)
        {
            // Unsupported type combination
            return nullptr;
        }
        else {
            return abs_contig_impl<argT, resT>;
        }
    }
};
```

## Memory Patterns

### Contiguous Arrays

```cpp
template <typename T>
void process_contig(sycl::queue &q,
                    size_t nelems,
                    const T *src,
                    T *dst,
                    const std::vector<sycl::event> &depends)
{
    sycl::event e = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for(sycl::range<1>(nelems),
                         [=](sycl::id<1> id) {
                             dst[id] = process(src[id]);
                         });
    });
}
```

### Strided Arrays

```cpp
template <typename T>
void process_strided(sycl::queue &q,
                     int nd,
                     const ssize_t *shape,
                     const T *src,
                     ssize_t src_offset,
                     const ssize_t *src_strides,
                     T *dst,
                     ssize_t dst_offset,
                     const ssize_t *dst_strides)
{
    // Use indexing utilities from offset_utils.hpp
    using dpctl::tensor::offset_utils::StridedIndexer;

    StridedIndexer src_indexer(nd, src_offset, shape, src_strides);
    StridedIndexer dst_indexer(nd, dst_offset, shape, dst_strides);

    // Submit kernel using indexers
}
```

## Common Utilities

### offset_utils.hpp
- `StridedIndexer` - Convert linear index to strided offset
- `UnpackedStridedIndexer` - Optimized for specific dimensions
- `TwoOffsets_StridedIndexer` - For binary operations

### sycl_alloc_utils.hpp
- `usm_allocator` - SYCL USM allocator wrapper
- Temporary allocation helpers

### type_utils.hpp
- `is_complex<T>` - Check if type is complex
- Type promotion utilities

## Best Practices

1. **Use `if constexpr`** for compile-time type branching
2. **Mark kernel classes** with unique names
3. **Support vectorization** where possible via `sycl::vec`
4. **Handle complex types** separately (no vectorization)
5. **Check device capabilities** before using fp64/fp16
6. **Use dispatch tables** for runtime type resolution
7. **Avoid raw new/delete** - use SYCL allocators
