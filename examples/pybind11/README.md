# Examples of data-parallel Python extensions written with pybind11

The `dpctl` provides integration header `dpctl4pybind11.hpp` which implements type casters
establishing mapping between `dpctl.SyclQueue` and `sycl::queue`, `dpctl.SyclDevice` and `sycl::device`,
`dpctl.SyclEvent` and `sycl::event`, etc.

The header also defines C++ classes `dpctl::tensor::usm_ndarray` and `dpctl::memory::usm_memory` which
derive from `pybind11::object` and encapsulate Python objects of types `dpctl.tensor.usm_ndarray` and
`dpctl.memory._Memory` respectively.
