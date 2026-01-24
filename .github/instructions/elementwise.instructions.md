---
applyTo:
  - "dpctl/tensor/libtensor/include/kernels/elementwise_functions/**"
  - "dpctl/tensor/libtensor/source/elementwise_functions/**"
  - "dpctl/tensor/_elementwise_funcs.py"
  - "dpctl/tensor/_elementwise_common.py"
  - "dpctl/tests/elementwise/**"
---

# Elementwise Operations Instructions

## Context

Elementwise operations span the full stack from C++ SYCL kernels to Python wrappers and tests. This guide covers the complete pattern for implementing elementwise operations.

## Full Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  dpctl/tensor/_elementwise_funcs.py                             │
│  Python wrapper using UnaryElementwiseFunc/BinaryElementwiseFunc│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  dpctl/tensor/_tensor_impl (pybind11)                           │
│  _abs, _abs_result_type, etc.                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  libtensor/source/elementwise_functions/abs.cpp                 │
│  Dispatch tables and pybind11 bindings                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  libtensor/include/kernels/elementwise_functions/abs.hpp        │
│  SYCL kernel implementation (AbsFunctor)                        │
└─────────────────────────────────────────────────────────────────┘
```

## Adding a New Elementwise Operation

### Step 1: C++ Kernel Header

Create `libtensor/include/kernels/elementwise_functions/myop.hpp`:

```cpp
//===-- myop.hpp - Implementation of myop             -*-C++-*-===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
// ... license ...
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "common.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::myop
{

namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

// Output type definition
template <typename argT, typename resT>
struct MyOpOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<argT, bool, bool>,
        td_ns::TypeMapResultEntry<argT, std::int8_t, std::int8_t>,
        // ... map all supported input types to output types
        td_ns::DefaultResultEntry<void>>::result_type;
};

// Kernel functor
template <typename argT, typename resT>
struct MyOpFunctor
{
    using supports_sg_loadstore =
        typename std::negation<tu_ns::is_complex<argT>>;
    using supports_vec =
        typename std::negation<tu_ns::is_complex<argT>>;

    resT operator()(const argT &x) const
    {
        if constexpr (tu_ns::is_complex<argT>::value) {
            // Handle complex types
            return /* complex implementation */;
        }
        else {
            return /* scalar implementation */;
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
    operator()(const sycl::vec<argT, vec_sz> &x) const
    {
        // Vectorized implementation
        return /* vectorized implementation */;
    }
};

// Factory for contiguous arrays
template <typename fnT, typename argT, typename resT>
struct MyOpContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                typename MyOpOutputType<argT, resT>::value_type, void>)
        {
            return nullptr;  // Unsupported type
        }
        else {
            using dpctl::tensor::kernels::unary_contig_impl;
            return unary_contig_impl<argT, resT, MyOpFunctor>;
        }
    }
};

// Factory for strided arrays
template <typename fnT, typename argT, typename resT>
struct MyOpStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                typename MyOpOutputType<argT, resT>::value_type, void>)
        {
            return nullptr;
        }
        else {
            using dpctl::tensor::kernels::unary_strided_impl;
            return unary_strided_impl<argT, resT, MyOpFunctor>;
        }
    }
};

// Type support check factory
template <typename fnT, typename argT, typename resT>
struct MyOpTypeSupportFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                typename MyOpOutputType<argT, resT>::value_type, void>)
        {
            return nullptr;
        }
        else {
            return [](void) -> bool { return true; };
        }
    }
};

} // namespace dpctl::tensor::kernels::myop
```

### Step 2: C++ Source File

Create `libtensor/source/elementwise_functions/myop.cpp`:

```cpp
//===----------- myop.cpp - Implementation of myop    -*-C++-*-===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
// ... license ...
//===----------------------------------------------------------------------===//

#include "myop.hpp"
#include "elementwise_functions.hpp"

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dpctl::tensor::py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

// Dispatch tables
static unary_contig_impl_fn_ptr_t
    myop_contig_dispatch_table[td_ns::num_types];

static unary_strided_impl_fn_ptr_t
    myop_strided_dispatch_table[td_ns::num_types];

void init_myop_dispatch_tables(void)
{
    using dpctl::tensor::kernels::myop::MyOpContigFactory;
    using dpctl::tensor::kernels::myop::MyOpStridedFactory;

    td_ns::DispatchTableBuilder<unary_contig_impl_fn_ptr_t,
                                MyOpContigFactory,
                                td_ns::num_types>
        dtb_contig;
    dtb_contig.populate_dispatch_table(myop_contig_dispatch_table);

    td_ns::DispatchTableBuilder<unary_strided_impl_fn_ptr_t,
                                MyOpStridedFactory,
                                td_ns::num_types>
        dtb_strided;
    dtb_strided.populate_dispatch_table(myop_strided_dispatch_table);
}

// Result type function
py::object myop_result_type(const py::dtype &input_dtype,
                            const py::object &device_obj)
{
    // Implementation using type lookup
}

// Main function
std::pair<sycl::event, sycl::event>
myop_func(const dpctl::tensor::usm_ndarray &src,
          const dpctl::tensor::usm_ndarray &dst,
          sycl::queue &exec_q,
          const std::vector<sycl::event> &depends)
{
    // Dispatch to appropriate kernel based on src dtype
}

void init_myop(py::module_ m)
{
    init_myop_dispatch_tables();

    m.def("_myop", &myop_func, "MyOp implementation",
          py::arg("src"), py::arg("dst"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());

    m.def("_myop_result_type", &myop_result_type);
}

} // namespace dpctl::tensor::py_internal
```

### Step 3: Register in tensor_elementwise.cpp

In `libtensor/source/tensor_elementwise.cpp`:

```cpp
#include "elementwise_functions/myop.hpp"

void init_elementwise_functions(py::module_ m)
{
    // ... existing operations ...
    init_myop(m);
}
```

### Step 4: Python Wrapper

In `dpctl/tensor/_elementwise_funcs.py`:

```python
import dpctl.tensor._tensor_impl as ti
from ._elementwise_common import UnaryElementwiseFunc

_myop_docstring_ = """
myop(x, /, out=None, order="K")

Computes myop element-wise.

Args:
    x (usm_ndarray): Input array.
    out (usm_ndarray, optional): Output array to use.
    order ({"K", "C", "F", "A"}, optional): Memory order for output.

Returns:
    usm_ndarray: Result array.

See Also:
    :func:`related_func`: Description.
"""

myop = UnaryElementwiseFunc(
    "myop",
    ti._myop_result_type,
    ti._myop,
    _myop_docstring_,
)
```

### Step 5: Export in __init__.py

In `dpctl/tensor/__init__.py`:

```python
from ._elementwise_funcs import (
    # ... existing exports ...
    myop,
)

__all__ = [
    # ... existing exports ...
    "myop",
]
```

### Step 6: Create Tests

Create `dpctl/tests/elementwise/test_myop.py`:

```python
import numpy as np
import pytest

import dpctl.tensor as dpt
from dpctl.tests.helper import get_queue_or_skip, skip_if_dtype_not_supported

from .utils import _all_dtypes, _usm_types


class TestMyOp:
    @pytest.mark.parametrize("dtype", _all_dtypes)
    def test_myop_dtype(self, dtype):
        q = get_queue_or_skip()
        skip_if_dtype_not_supported(dtype, q)

        x = dpt.linspace(0, 1, 100, dtype=dtype, sycl_queue=q)
        result = dpt.myop(x)

        expected = np.myop(dpt.asnumpy(x))
        assert np.allclose(dpt.asnumpy(result), expected)

    @pytest.mark.parametrize("usm_type", _usm_types)
    def test_myop_usm_type(self, usm_type):
        q = get_queue_or_skip()

        x = dpt.ones(100, dtype="f4", usm_type=usm_type, sycl_queue=q)
        result = dpt.myop(x)

        assert result.usm_type == usm_type

    def test_myop_empty(self):
        q = get_queue_or_skip()
        x = dpt.empty((0,), dtype="f4", sycl_queue=q)
        result = dpt.myop(x)
        assert result.shape == (0,)

    def test_myop_scalar(self):
        q = get_queue_or_skip()
        x = dpt.asarray(5.0, sycl_queue=q)
        result = dpt.myop(x)
        assert result.ndim == 0

    def test_myop_out(self):
        q = get_queue_or_skip()
        x = dpt.ones(100, dtype="f4", sycl_queue=q)
        out = dpt.empty_like(x)
        result = dpt.myop(x, out=out)
        assert result is out

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_myop_order(self, order):
        q = get_queue_or_skip()
        x = dpt.ones((10, 10), dtype="f4", order=order, sycl_queue=q)
        result = dpt.myop(x, order=order)

        if order == "C":
            assert result.flags.c_contiguous
        else:
            assert result.flags.f_contiguous
```

## Type Support Matrix

Common type mappings for unary operations:

| Operation | Input Types | Output Type |
|-----------|-------------|-------------|
| abs | int, float | same |
| abs | complex | real component type |
| negative | int, float, complex | same |
| sin, cos, etc. | float, complex | same |
| isnan, isinf | float, complex | bool |
| sign | int, float, complex | same |

## Binary Operation Pattern

For binary operations, use `BinaryElementwiseFunc`:

```python
from ._elementwise_common import BinaryElementwiseFunc

add = BinaryElementwiseFunc(
    "add",
    ti._add_result_type,
    ti._add,
    _add_docstring_,
    # Binary operations may need additional parameters
    ti._add_inplace,  # In-place variant
)
```

## Vectorization Notes

- `supports_sg_loadstore` - Can use sub-group load/store
- `supports_vec` - Can use sycl::vec for vectorization
- Complex types typically don't support vectorization
- Check device capabilities for half-precision (fp16)
