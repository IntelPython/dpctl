# Examples of data-parallel Python extensions written with pybind11

The `dpctl` provides integration header `dpctl4pybind11.hpp` which implements type casters
establishing mapping between `dpctl.SyclQueue` and `sycl::queue`, `dpctl.SyclDevice` and `sycl::device`,
`dpctl.SyclEvent` and `sycl::event`, etc.

The header also defines C++ class `dpctl::memory::usm_memory` which derives from `pybind11::object` and encapsulates Python objects of type `dpctl.memory._Memory`.
