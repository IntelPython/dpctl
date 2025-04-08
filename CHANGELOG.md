# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [dev] - XXX. XX, XXXX

### Added

* Added `out` keyword to `tensor.take` [gh-2010](https://github.com/IntelPython/dpctl/pull/2010)

### Changed

* Support for Boolean data-type is added to `dpctl.tensor.ceil`, `dpctl.tensor.floor`, and `dpctl.tensor.trunc` [gh-2033](https://github.com/IntelPython/dpctl/pull/2033)
* Changed implementation of `DPCTLPlatform_GetDefaultContext` from using deprecated `ext_oneapi_get_default_context` to `khr_get_default_context` [#2042](https://github.com/IntelPython/dpctl/pull/2042).

### Fixed

## [0.19.0] - Feb. 26, 2025

This release features official, out-of-the-box support for compiling `dpctl` for specified AMD GPU architectures, the addition of new function `tensor.top_k`, a radix-sort-based implementation of sorting functions, and improvements to interoperability with DLPack through `tensor.dldevice_to_sycl_device` and `tensor.sycl_device_to_dldevice`.

A number of adjustments were also made to improve performance of `dpctl` reductions (i.e., `sum`, `min`, `max`, etc.), accumulators (i.e., `cumulative_sum`, `cumulative_logsumexp`), and copy-and-cast operations.

### Added

* Support for compiling `dpctl` for specified AMD GPU architecture with use of [CodePlay oneAPI plug-in](https://developer.codeplay.com/products/oneapi/amd/home/) [gh-1731](https://github.com/IntelPython/dpctl/pull/1731)
* Added `tensor.top_k` per Python Array API specification [gh-1921](https://github.com/IntelPython/dpctl/pull/1921)
* Added functions `tensor.dldevice_to_sycl_device` and `tensor.sycl_device_to_dldevice` for converting between DLPack and sycl devices, and a method `get_device_id` to `dpctl.SyclDevice` to improve interoperability with DLPack protocol [gh-1953](https://github.com/IntelPython/dpctl/pull/1953)
* Added `DPCTL_OFFLOAD_COMPRESS` cmake option (set to `OFF` by default) to toggle [--offload-compress](https://www.intel.com/content/www/us/en/developer/articles/technical/sycl-compilation-device-image-compression.html) linker option when building `dpctl` [gh-1961](https://github.com/IntelPython/dpctl/pull/1961)

### Changed

* Improved performance of copy-and-cast operations from `numpy.ndarray` to `tensor.usm_ndarray` for contiguous inputs [gh-1829](https://github.com/IntelPython/dpctl/pull/1829)
* `py_sort` and `py_argsort` now throw `py::value_error` if inputs are not C-contiguous [gh-1838](https://github.com/IntelPython/dpctl/pull/1838)
* Improved performance of copying operation to C-/F-contig array, with optimization for batch of square matrices [gh-1850](https://github.com/IntelPython/dpctl/pull/1850)
* Improved performance of `tensor.argsort` function for all types [gh-1859](https://github.com/IntelPython/dpctl/pull/1859)
* Improved performance of `tensor.sort` and `tensor.argsort` for short arrays in the range [16, 64] elements [gh-1866](https://github.com/IntelPython/dpctl/pull/1866)
* Implemented radix sort algorithm to be used in `dpt.sort` and `dpt.argsort` [gh-1867](https://github.com/IntelPython/dpctl/pull/1867), [gh-1883](https://github.com/IntelPython/dpctl/pull/1883)
* Extended `dpctl.SyclTimer` with `device_timer` keyword, implementing different methods of collecting device times [gh-1872](https://github.com/IntelPython/dpctl/pull/1872)
* `dpctl` changed to see GPU devices out of the box in virtual environment on Windows [gh-1922](https://github.com/IntelPython/dpctl/pull/1922)
* Improved performance of `tensor.cumulative_sum`, `tensor.cumulative_prod`, `tensor.cumulative_logsumexp` as well as performance of boolean indexing [gh-1923](https://github.com/IntelPython/dpctl/pull/1923), [gh-1942](https://github.com/IntelPython/dpctl/pull/1942)
* Improved performance of `tensor.min`, `tensor.max`, `tensor.logsumexp`, `tensor.reduce_hypot` for floating point type arrays by at least 2x [gh-1932](https://github.com/IntelPython/dpctl/pull/1932), [gh-1937](https://github.com/IntelPython/dpctl/pull/1937)
* Updated Cython examples to use scikit-build [gh-1935](https://github.com/IntelPython/dpctl/pull/1935)
* Reduced binary size of `_tensor_accumulation_impl` by 13 MB [gh-1957](https://github.com/IntelPython/dpctl/pull/1957)
* Extended `tensor.asarray` to support objects that implement `__usm_ndarray__` property to be interpreted as `usm_ndarray` objects [gh-1959](https://github.com/IntelPython/dpctl/pull/1959)
* `tensor.usm_ndarray` object disallows implicit conversions to NumPy array [gh-1964](https://github.com/IntelPython/dpctl/pull/1964)
* `stream` arguments in `tensor.usm_ndarray` methods now raise an error if `stream` is not a `tensor.SyclQueue` [gh-1969](https://github.com/IntelPython/dpctl/pull/1969)
* `dpctl` initialization sets subprocess to use SPAWN method on Linux to enable `gdb-oneapi` to debug kernels submitted from Python applications [gh-1971](https://github.com/IntelPython/dpctl/pull/1971)
* Reduced binary size of `_tensor_elementwise_impl` [gh-1976](https://github.com/IntelPython/dpctl/pull/1976)
* Allow `dpctl.SyclQueue.memcpy` to and from multi-dimensional buffers [gh-1985](https://github.com/IntelPython/dpctl/pull/1985)

### Fixed

* Fixed a bug in `tensor.roll` for very large values of `shift` [gh-1869](https://github.com/IntelPython/dpctl/pull/1869)
* Fix for `tensor.result_type` when all inputs are Python built-in scalars [gh-1877](https://github.com/IntelPython/dpctl/pull/1877)
* Improved error in constructors `tensor.full` and `tensor.full_like` when provided a non-numeric fill value [gh-1878](https://github.com/IntelPython/dpctl/pull/1878)
* Added a check for pointer alignment when copying to C-contiguous memory [gh-1890](https://github.com/IntelPython/dpctl/pull/1890), [gh-1891](https://github.com/IntelPython/dpctl/pull/1891)
* Fixed `dpctl` installed into virtual environment not finding DPC++ runtime libraries by adding `DPCTL_WITH_REDIST` cmake option (set to `OFF` by default) [gh-1893](https://github.com/IntelPython/dpctl/pull/1893)
* Fixed incorrect result (issue [gh-1901](https://github.com/IntelPython/dpctl/issues/1901)) in `tensor.cumulative_sum` and in advanced indexing [gh-1902](https://github.com/IntelPython/dpctl/pull/1902)
* Fixed `__setitem__()` for `tensor.usm_ndarray` when passed an empty boolean mask [gh-1915](https://github.com/IntelPython/dpctl/pull/1915)
* `tensor.from_dlpack` docstring now shows that return type can be NumPy array and stipulates when this will be the case [gh-1919](https://github.com/IntelPython/dpctl/pull/1919)
* Fixed docstring in helper class in DLPack tests [gh-1920](https://github.com/IntelPython/dpctl/pull/1920)
* Fixed a bug in `tensor.astype` where `copy=False` would not be respected for 1d arrays when order keyword is specified [gh-1928](https://github.com/IntelPython/dpctl/pull/1928)
* Replaced deprecated `CL/sycl.hpp` with recommended `sycl/sycl.hpp` in examples [gh-1933](https://github.com/IntelPython/dpctl/pull/1933)
* Fixed `tensor.take_along_axis` and `tensor.put_along_axis` raising an error for `tensor.uint64` indices when given an array of dimension greater than 1 [gh-1934](https://github.com/IntelPython/dpctl/pull/1934)
* Fixed unexpected results of `tensor.sum` with a requested output type of `bool` [gh-1958](https://github.com/IntelPython/dpctl/pull/1958)
* Use `std::move` to avoid unnecessary copying of temporary in `triul_ctor.cpp` [gh-1960](https://github.com/IntelPython/dpctl/pull/1960)
* Make `stream` a keyword-only argument in `tensor.usm_ndarray.to_device` per requirement by array API specification [gh-1966](https://github.com/IntelPython/dpctl/pull/1966)
* Improve efficiency of copy implementation and avoid an unnecessary kernel invocation in `tensor.argsort` for 1d input [gh-1967](https://github.com/IntelPython/dpctl/pull/1967)
* Corrected uses of NumPy constructors with `tensor.usm_ndarray` inputs in test suite [gh-1968](https://github.com/IntelPython/dpctl/pull/1968)
* Fixed array API namespace inspection utilities showing `complex128` as a valid dtype on devices without double precision and `device` keywords not working with `dpctl.SyclQueue` or filter strings [gh-1979](https://github.com/IntelPython/dpctl/pull/1979)
* Fixed a bug in `test_sycl_device_interface.cpp` which would cause compilation to fail with Clang version 20.0 [gh-1989](https://github.com/IntelPython/dpctl/pull/1989)
* Fixed memory leaks in smart-pointer-managed USM temporaries in synchronizing kernel calls [gh-2002](https://github.com/IntelPython/dpctl/pull/2002)
* `UsmNDArray_MakeSimpleFromPtr` and `UsmNDArray_MakeFromPtr` now raise an error when provided an invalid `typenum` before attempting to create the array [gh-2003](https://github.com/IntelPython/dpctl/pull/2003)
* Fixed typos in `tensor.from_numpy` and `tensor.astype` [gh-2006](https://github.com/IntelPython/dpctl/pull/2006)

### Maintenance

* Revert pinning of cmake to 3.26 on Windows [gh-1823](https://github.com/IntelPython/dpctl/pull/1823)
* Update black version used in Python code style workflow [gh-1828](https://github.com/IntelPython/dpctl/pull/1828)
* Fixed CI/CD workflow for building conda packages on Windows [gh-1831](https://github.com/IntelPython/dpctl/pull/1831)
* Revert work-around in `test_sycl_kernel_submit.py` for problem in MKL 2024.2.0 [gh-1836](https://github.com/IntelPython/dpctl/pull/1836)
* Do not use Mambaforge variant of miniforge as deprecated [gh-1844](https://github.com/IntelPython/dpctl/pull/1844)
* Use pybind11=2.13.6 [gh-1845](https://github.com/IntelPython/dpctl/pull/1845)
* Remove unnecessary include in C++ header file [gh-1846](https://github.com/IntelPython/dpctl/pull/1846)
* Build translation unit "simplify_iteration_space.cpp" compiled multiple times as a static library [gh-1847](https://github.com/IntelPython/dpctl/pull/1847)
* Add instructions for installing `dpctl` from Intel PyPi channel [gh-1860](https://github.com/IntelPython/dpctl/pull/1860)
* Fix warnings when generating docs [gh-1855](https://github.com/IntelPython/dpctl/pull/1855), [gh-1861](https://github.com/IntelPython/dpctl/pull/1861)
* Align conda recipe with conda-forge's `{{ stdlib("c") }}` migration [gh-1868](https://github.com/IntelPython/dpctl/pull/1868)
* Add missing include of SYCL header to "math_utils.hpp" [gh-1899](https://github.com/IntelPython/dpctl/pull/1899)
* Add support of CV-qualifiers in `is_complex<T>` helper [gh-1900](https://github.com/IntelPython/dpctl/pull/1900)
* Tuning work for elementwise functions with modest performance gains (under 10%) [gh-1889](https://github.com/IntelPython/dpctl/pull/1889)
* Reduce binary size of accumulators by saving repeated expressions to a temporary [gh-1896](https://github.com/IntelPython/dpctl/pull/1896)
* Added workflow to run nightly tests of `dpctl` [gh-1903](https://github.com/IntelPython/dpctl/pull/1903), [gh-1905](https://github.com/IntelPython/dpctl/pull/1905)
* Support and testing for Python 3.13 for `dpctl` [gh-1941](https://github.com/IntelPython/dpctl/pull/1941), [gh-1943](https://github.com/IntelPython/dpctl/pull/1943)
* Change libtensor to use `std::size_t` and `dpctl::tensor::ssize_t` throughout and fix missing includes for `std::size_t` and `size_t` [gh-1950](https://github.com/IntelPython/dpctl/pull/1950)
* Fixed some unqualified `size_t` and fixed-width integral types in `libtensor` [gh-1955](https://github.com/IntelPython/dpctl/pull/1955)
* Add versioneer as a build requirement in documentation on building `dpctl` from source [gh-1972](https://github.com/IntelPython/dpctl/pull/1972)
* Remove const qualifiers for class and struct members [gh-1974](https://github.com/IntelPython/dpctl/pull/1974), [gh-1975](https://github.com/IntelPython/dpctl/pull/1975)
* Various code quality improvements to `test_sycl_queue_submit_local_accessor_arg.cpp` [gh-1990](https://github.com/IntelPython/dpctl/pull/1990)
* Added Python 3.12 to package metadata [gh-2005](https://github.com/IntelPython/dpctl/pull/2005)
* Miscellaneous changes to continuous integration/delivery (CI/CD) supporting scripts:
[gh-1837](https://github.com/IntelPython/dpctl/pull/1837),
[gh-1839](https://github.com/IntelPython/dpctl/pull/1839),
[gh-1848](https://github.com/IntelPython/dpctl/pull/1848),
[gh-1853](https://github.com/IntelPython/dpctl/pull/1853),
[gh-1854](https://github.com/IntelPython/dpctl/pull/1854),
[gh-1856](https://github.com/IntelPython/dpctl/pull/1856),
[gh-1858](https://github.com/IntelPython/dpctl/pull/1858),
[gh-1863](https://github.com/IntelPython/dpctl/pull/1863),
[gh-1864](https://github.com/IntelPython/dpctl/pull/1864),
[gh-1865](https://github.com/IntelPython/dpctl/pull/1865),
[gh-1881](https://github.com/IntelPython/dpctl/pull/1881),
[gh-1882](https://github.com/IntelPython/dpctl/pull/1882),
[gh-1884](https://github.com/IntelPython/dpctl/pull/1884),
[gh-1886](https://github.com/IntelPython/dpctl/pull/1886),
[gh-1888](https://github.com/IntelPython/dpctl/pull/1888),
[gh-1897](https://github.com/IntelPython/dpctl/pull/1897),
[gh-1898](https://github.com/IntelPython/dpctl/pull/1898),
[gh-1909](https://github.com/IntelPython/dpctl/pull/1909),
[gh-1916](https://github.com/IntelPython/dpctl/pull/1916),
[gh-1927](https://github.com/IntelPython/dpctl/pull/1927),
[gh-1940](https://github.com/IntelPython/dpctl/pull/1940),
[gh-1948](https://github.com/IntelPython/dpctl/pull/1948),
[gh-1949](https://github.com/IntelPython/dpctl/pull/1949),
[gh-1952](https://github.com/IntelPython/dpctl/pull/1952),
[gh-1962](https://github.com/IntelPython/dpctl/pull/1962),
[gh-1963](https://github.com/IntelPython/dpctl/pull/1963),
[gh-1973](https://github.com/IntelPython/dpctl/pull/1973),
[gh-1980](https://github.com/IntelPython/dpctl/pull/1980),
[gh-1981](https://github.com/IntelPython/dpctl/pull/1981),
[gh-1983](https://github.com/IntelPython/dpctl/pull/1983),
[gh-1988](https://github.com/IntelPython/dpctl/pull/1988),

## [0.18.3] - Dec. 07, 2024

### Fixed

* Enabled `dpctl` in virtual environment on Windows platform (issue [gh-1745](https://github.com/IntelPython/dpctl/issues/1745)) [gh-1924](https://github.com/IntelPython/dpctl/pull/1924)

## [0.18.2] - Nov. 21, 2024

### Maintenance

* Add missing include of SYCL header to "math_utils.hpp" [gh-1899](https://github.com/IntelPython/dpctl/pull/1899)

### Fixed

* Fix for `tensor.result_type` when all inputs are Python built-in scalars [gh-1904](https://github.com/IntelPython/dpctl/pull/1904)

## [0.18.1] - Oct. 11, 2024

### Changed

* Updated installation instructions [gh-1862](https://github.com/IntelPython/dpctl/pull/1862)

## [0.18.0] - Sept. 26, 2024

This release reaches an important milestone by making offloading fully asynchronous.
Calls to `dpctl.tensor` submit tasks for execution to DPC++ runtime and return without waiting for execution of these tasks to finish.
The sequential semantics a user comes to expect from execution of Python script is preserved though.

The full list of changes that went into this release are:

### Added

* Implement `tensor.take_along_axis` per Python Array API specification [gh-1778](https://github.com/IntelPython/dpctl/pull/1778)
* Implement `tensor.put_along_axis` to complement `tensor.take_along_axis` [gh-1798](https://github.com/IntelPython/dpctl/pull/1798)
* Support for 'device=tensor.kDLCPU' in `tensor.from_dlpack` function and `tensor.usm_ndarray.__dlpack__` method [gh-1781](https://github.com/IntelPython/dpctl/pull/1781)
* Support DLPack on Windows [gh-1746](https://github.com/IntelPython/dpctl/pull/1746)
* Implement `tensor.nextafter` function per Python Array API specification [gh-1730](https://github.com/IntelPython/dpctl/pull/1730)
* Implement `tensor.count_nonzero` and `tensor.diff` functions from Python array API specification [gh-1732](https://github.com/IntelPython/dpctl/pull/1732), [gh-1780](https://github.com/IntelPython/dpctl/pull/1780)
* Add support for `order="K"` to `*_like` array creation functions, and change default `order` keyword value from `'C'` to `'K'` [gh-1808](https://github.com/IntelPython/dpctl/pull/1808)
* Support for 'max dimensions' in Array API capabilities info data [gh-1774](https://github.com/IntelPython/dpctl/pull/1774)
* Add support for device aspect 'emulated' [gh-1691](https://github.com/IntelPython/dpctl/pull/1691)
* `dpctl::tensor::usm_memory` class defined in `dpctl4pybind11.hpp` adds constructor to create Python USM memory objects viewing into existing USM allocations, which can be made by an external library [gh-1782](https://github.com/IntelPython/dpctl/pull/1782)
* Add support for COVERAGE build type in project's CMake script [gh-1692](https://github.com/IntelPython/dpctl/pull/1692)

### Changed

* Change ownership of USM allocation by `dpctl.memory` objects, make executions of `dpctl.tensor` operations asynchronous [gh-1705](https://github.com/IntelPython/dpctl/pull/1705)
* Add support for Python scalars by `tensor.where` function [gh-1719](https://github.com/IntelPython/dpctl/pull/1719)
* Optimize division by Python scalar in statistical functions `tensor.mean`, `tensor.std`, `tensor.var` [gh-1820](https://github.com/IntelPython/dpctl/pull/1820)
* Use transcendental functions from `sycl` namespace instead of `std` namespace [gh-1707](https://github.com/IntelPython/dpctl/pull/1707)
* Changes for compatibility with recent NumPy in runtime environment [gh-1735](https://github.com/IntelPython/dpctl/pull/1735), [gh-1772](https://github.com/IntelPython/dpctl/pull/1772), [gh-1804](https://github.com/IntelPython/dpctl/pull/1804)
* Array creation function `tensor.zeros` to use asynchronous `memset` operation [gh-1806](https://github.com/IntelPython/dpctl/pull/1806)
* The setter of `tensor.usm_ndarray.shape` property now supports Python scalar value [gh-1786](https://github.com/IntelPython/dpctl/pull/1786)
* Use 'pyproject.toml' instead of 'setup.py' aligning with current packaging best practices [gh-1660](https://github.com/IntelPython/dpctl/pull/1660)
* No longer set SOVERSION property in DPCTLSyclInterface library on Linux [gh-1773](https://github.com/IntelPython/dpctl/pull/1773)
* Update version of 'pybind11' used [gh-1758](https://github.com/IntelPython/dpctl/pull/1758), [gh-1812](https://github.com/IntelPython/dpctl/pull/1812)
* Handle possible exceptions by `usm_host_allocator` used with `std::vector` [gh-1791](https://github.com/IntelPython/dpctl/pull/1791)
* Use `dpctl::tensor::alloc_utils::sycl_free_noexcept` instead of `sycl::free` in `host_task` tasks associated with life-time management of temporary USM allocations [gh-1797](https://github.com/IntelPython/dpctl/pull/1797)
* Add `"same_kind"`-style casting for in-place mathematical operators of `tensor.usm_ndarray` [gh-1827](https://github.com/IntelPython/dpctl/pull/1827), [gh-1830](https://github.com/IntelPython/dpctl/pull/1830)

### Fixed

* Fix setting of release variable Sphinx config file [gh-1685](https://github.com/IntelPython/dpctl/pull/1685)
* Handle possible NULL return value from device aspect queries `DPCTLDevice_GetMaxWorkGroupSize1d` and `DPCTLDevice_GetMaxWorkGroupSize2d` [gh-1690](https://github.com/IntelPython/dpctl/pull/1690)
* Add license header to conda script files [gh-1695](https://github.com/IntelPython/dpctl/pull/1695)
* Fix `tensor.round` behavior on CUDA devices [gh-1700](https://github.com/IntelPython/dpctl/pull/1700)
* Add missing `#include <sstream>` [gh-1701](https://github.com/IntelPython/dpctl/pull/1701)
* Fix for issue 1724 [gh-1728](https://github.com/IntelPython/dpctl/pull/1728)
* Correct USM type for return array of `tensor.extract` function [gh-1727](https://github.com/IntelPython/dpctl/pull/1727)
* Fix for `tensor.unique_all` and `tensor.unique_inverse` to always return index arrays with default indexing data type [gh-1741](https://github.com/IntelPython/dpctl/pull/1741)
* Propagate read-only flag from `__sycl_usm_array_interface__` in `tensor.asarray` function [gh-1756](https://github.com/IntelPython/dpctl/pull/1756)
* `tensor.clip` to handle Python scalars which are out of bound for the data type of integral array [gh-1759](https://github.com/IntelPython/dpctl/pull/1759)
* Avoid dead-locking by releasing GIL around blocking operations in libtensor [gh-1753](https://github.com/IntelPython/dpctl/pull/1753)
* Element-wise `tensor.divide` and comparison operations allow greater range of Python integer and integer array combinations [gh-1771](https://github.com/IntelPython/dpctl/pull/1771)
* Fix for unexpected behavior when using floating point types for array indexing [gh-1792](https://github.com/IntelPython/dpctl/pull/1792)
* Enable `pytest --pyargs dpctl.tests` [gh-1833](https://github.com/IntelPython/dpctl/pull/1833)
* Fix for undefined behavior in indexing using integer arrays [gh-1894](https://github.com/IntelPython/dpctl/pull/1894)

### Maintenance

* Improve performance of `test_sort_complex_fp_nan` [gh-1704](https://github.com/IntelPython/dpctl/pull/1704)
* Improve exception wording raised by `tensor.broadcast_arrays()` [gh-1720](https://github.com/IntelPython/dpctl/pull/1720)
* Remove `template` keyword in method call of `sycl::kernel_bundle` [gh-1726](https://github.com/IntelPython/dpctl/pull/1726)
* Backport changelog edits from maintenance/0.17.x [gh-1736](https://github.com/IntelPython/dpctl/pull/1736)
* Replace uses of 'intel' channels in docs and readme file [gh-1737](https://github.com/IntelPython/dpctl/pull/1737)
* Update references to deprecated environment variable `SYCL_DEVICE_FILTER` [gh-1740](https://github.com/IntelPython/dpctl/pull/1740)
* Correction for installation instruction steps [gh-1754](https://github.com/IntelPython/dpctl/pull/1754)
* Fix for crash during testing with open source SYCL bundle by updating CPU RT library used [gh-1762](https://github.com/IntelPython/dpctl/pull/1762)
* Add missing include to fix build break with newer LLVM [gh-1776](https://github.com/IntelPython/dpctl/pull/1776)
* Add `#include <utility>` for definition of `std::move` used [gh-1787](https://github.com/IntelPython/dpctl/pull/1787)
* Change to CMake script to accomodate DPC++ transition from PI to UR architecture [gh-1788](https://github.com/IntelPython/dpctl/pull/1788)
* Document `tensor._flags.Flags` class [gh-1794](https://github.com/IntelPython/dpctl/pull/1794)
* Fix for unreferenced unreleased bug in copy-and-cast code logic [gh-1799](https://github.com/IntelPython/dpctl/pull/1799)
* Explicitly include headers used in C++ translation units implementing reduction operations [gh-1802](https://github.com/IntelPython/dpctl/pull/1802)
* Clean-up uses of `Strided1DIndexer` class [gh-1805](https://github.com/IntelPython/dpctl/pull/1805)
* Tweak to readability of C++ code implementing matrix-matrix multiplication [gh-1810](https://github.com/IntelPython/dpctl/pull/1810)
* Do not add `sycl::event` associated with compute task to vector of events representing execution of `host_task` [gh-1807](https://github.com/IntelPython/dpctl/pull/1807)
* Remove 'level-zero' conda package from run-time dependencies of 'dpctl' since Intel GPU driver stack now explicitly depends on `libze1` package which provides Level-Zero loader library [gh-1801](https://github.com/IntelPython/dpctl/pull/1801), [gh-1840](https://github.com/IntelPython/dpctl/pull/1840)
* Use dedicated type-support matrices for in-place element-wise binary operations [gh-1816](https://github.com/IntelPython/dpctl/pull/1816)
* Remove recommendation to install wheels from Anaconda PyPI index [gh-1819](https://github.com/IntelPython/dpctl/pull/1819)
* Removed use of post-link and pre-unlink conda scripts in `dpctl` [gh-1821](https://github.com/IntelPython/dpctl/pull/1821)
* Pin compiler used to build 0.18.0 version to 2025.0.0 [gh-1822](https://github.com/IntelPython/dpctl/pull/1822)
* A varienty of changes to continuous integration/delivery (CI/CD) supporting scripts to keep CI running smoothly:
 [gh-1686](https://github.com/IntelPython/dpctl/pull/1686),
 [gh-1688](https://github.com/IntelPython/dpctl/pull/1688),
 [gh-1697](https://github.com/IntelPython/dpctl/pull/1697),
 [gh-1698](https://github.com/IntelPython/dpctl/pull/1698),
 [gh-1703](https://github.com/IntelPython/dpctl/pull/1703),
 [gh-1702](https://github.com/IntelPython/dpctl/pull/1702),
 [gh-1709](https://github.com/IntelPython/dpctl/pull/1709),
 [gh-1712](https://github.com/IntelPython/dpctl/pull/1712),
 [gh-1713](https://github.com/IntelPython/dpctl/pull/1713),
 [gh-1722](https://github.com/IntelPython/dpctl/pull/1722),
 [gh-1725](https://github.com/IntelPython/dpctl/pull/1725),
 [gh-1729](https://github.com/IntelPython/dpctl/pull/1729),
 [gh-1733](https://github.com/IntelPython/dpctl/pull/1733),
 [gh-1721](https://github.com/IntelPython/dpctl/pull/1721),
 [gh-1743](https://github.com/IntelPython/dpctl/pull/1743),
 [gh-1739](https://github.com/IntelPython/dpctl/pull/1739),
 [gh-1747](https://github.com/IntelPython/dpctl/pull/1747),
 [gh-1748](https://github.com/IntelPython/dpctl/pull/1748),
 [gh-1750](https://github.com/IntelPython/dpctl/pull/1750),
 [gh-1752](https://github.com/IntelPython/dpctl/pull/1752),
 [gh-1767](https://github.com/IntelPython/dpctl/pull/1767),
 [gh-1768](https://github.com/IntelPython/dpctl/pull/1768),
 [gh-1775](https://github.com/IntelPython/dpctl/pull/1775),
 [gh-1783](https://github.com/IntelPython/dpctl/pull/1783),
 [gh-1790](https://github.com/IntelPython/dpctl/pull/1790),
 [gh-1795](https://github.com/IntelPython/dpctl/pull/1795),
 [gh-1796](https://github.com/IntelPython/dpctl/pull/1796),
 [gh-1800](https://github.com/IntelPython/dpctl/pull/1800),
 [gh-1760](https://github.com/IntelPython/dpctl/pull/1760),
 [gh-1803](https://github.com/IntelPython/dpctl/pull/1803),
 [gh-1777](https://github.com/IntelPython/dpctl/pull/1777),
 [gh-1813](https://github.com/IntelPython/dpctl/pull/1813),
 [gh-1817](https://github.com/IntelPython/dpctl/pull/1817),
 [gh-1818](https://github.com/IntelPython/dpctl/pull/1818)

## [0.17.0] - May. 23, 2024

This release features updated documentation web-page https://intelpython.github.io/dpctl/latest/index.html, adds cumulative reductions,
and complies with revision [2023.12](https://data-apis.org/array-api/2023.12/) of Python Array API specification.

### Added

* Added pybind11 caster for ``sycl::half`` to map to/from Python `float` to ``"dpctl4pybind11.hpp"`` header: [gh-1655](https://github.com/IntelPython/dpctl/pull/1655)
* Added support for DLPack data interchange per Python Array API 2023.12 specification: [gh-1667](https://github.com/IntelPython/dpctl/pull/1667)
* Implemented `tensor.cumulative_sum`, `tensor.cumulative_prod` and `tensor.cumulative_logsumexp`: [gh-1602](https://github.com/IntelPython/dpctl/pull/1602)

### Changed

* Expanded documentation for `dpctl`: [gh-1619](https://github.com/IntelPython/dpctl/pull/1619)
* Expanded `utils.intel_device_info` functionality: [gh-1656](https://github.com/IntelPython/dpctl/pull/1656)
* Improved performance of elementwise operations: [gh-1651](https://github.com/IntelPython/dpctl/pull/1651)
* Efficiency improvement by avoiding unnecessary copying of ``sycl::queue``: [gh-1645](https://github.com/IntelPython/dpctl/pull/1645)
* `dpctl` uses pybind11 2.12.0: [gh-1640](https://github.com/IntelPython/dpctl/pull/1640)
* Improved performance of `tensor.reshape` operation with `order="F"` when copying is needed, or requested: [gh-1677](https://github.com/IntelPython/dpctl/pull/1677)

### Fixed

* Fixed initialization of byte type constants in `dpctl_capi` Python/C API loader class in `"dpctl4pybind11.hpp"`: [gh-1665](https://github.com/IntelPython/dpctl/pull/1665)
* Fixed crash in `tensor.sort` reported for a CPU device and a CUDA device: [gh-1676](https://github.com/IntelPython/dpctl/pull/1676)
* Fixed race condition in accumulation kernel for custom operations that caused test failures with AMD CPUs: [gh-1624](https://github.com/IntelPython/dpctl/pull/1624)
* Fixed comparison operators for mixed signed and unsigned integral types: [gh-1650](https://github.com/IntelPython/dpctl/pull/1650)
* Support use of index arrays of different integral types in indexing operations: [gh-47](https://github.com/IntelPython/dpctl/pull/1647)
* Fixed source code to compile for NVidia(TM) GPUs with DPC++ 2024.1: [gh-1630](https://github.com/IntelPython/dpctl/pull/1630)
* Corrected `tensor.tile` for scalar inputs and empty repetitions: [gh-1628](https://github.com/IntelPython/dpctl/pull/1628)
* Fixed support for `out` keyword in `tensor.matmul`: [gh-1610](https://github.com/IntelPython/dpctl/pull/1610)
* Fixed bug in basic slicing of empty arrays: [gh-1680](https://github.com/IntelPython/dpctl/pull/1680)
* Fixed bug in `tensor.bitwise_invert` for boolean input array: [gh-1681](https://github.com/IntelPython/dpctl/pull/1681)
* Fixed bug in `tensor.repeat` on zero-size input arrays: [gh-1682](https://github.com/IntelPython/dpctl/pull/1682)
* Fixed bug in `tensor.searchsorted` for 0d needle vector and strided hay: [gh-1694](https://github.com/IntelPython/dpctl/pull/1694)


## [0.16.1] - Apr. 10, 2024

This is a bug-fix release, which also provides a change needed by ``numba_dpex`` project to support dispatching kernels
consuming instances of ``sycl::local_accessor`` template type.

### Changed

* Changed behavior of ``dpctl.tensor.usm_ndarray.__dlpack_device__`` method to return device id of the parent unpartitioned device if array is allocated on a sub-device instead of raising an exception: [#1604](https://github.com/IntelPython/dpctl/pull/1604)
* Array creation functions and the ``usm_ndarray`` constructor in `dpctl.tensor` submodule now use cached default-selected device to improve performance: [#1606](https://github.com/IntelPython/dpctl/pull/1606)
* Changed treatment of `axis` keyword for `dpctl.tensor.tensordot` and `dpctl.tensor.vecdot` to align with Python Array API 2023.12 specification: [#1608](https://github.com/IntelPython/dpctl/pull/1608)
* Changed implementation of `DPCTLQueue_SubmitRange`, `DPCTLQueue_SubmitNDRange` in DPCTLSyclInterface library to support ``sycl::local_accessor`` arguments needed by ``numba_dpex``; the enum `DPCTLKernelArgType` to correspond to C++ disjoint types: [#1609](https://github.com/IntelPython/dpctl/pull/1609), [#1611](https://github.com/IntelPython/dpctl/pull/1611), [#1612](https://github.com/IntelPython/dpctl/pull/1612)

### Fixed

* Fixed a crash on Windows platform during execution of getter of `dpctl.SyclPlatfom.default_context` property: : [#1604](https://github.com/IntelPython/dpctl/pull/1604)
* Fixed kernel submission error on NVidia CUDA GPUs during `dpctl.tensor.matmul` operation: [#1605](https://github.com/IntelPython/dpctl/pull/1605)
* Fixed corruption of context cache table entries: [#1607](https://github.com/IntelPython/dpctl/pull/1607)
* Fixed incorrect result from ``dpctl.tensor.tensordot`` reported in issue [#1570](https://github.com/IntelPython/dpctl/issues/1570): [#1608](https://github.com/IntelPython/dpctl/pull/1608)
* Fixed library name output by ``python -m dpctl --library``: [#1615](https://github.com/IntelPython/dpctl/pull/1615)


## [0.16.0] - Feb. 16, 2024

This release will require DPC++ 2024.1.0, which no longer supports Intel Gen9 integrated GPUs found in Intel CPUs of 10th generation and older.
Featurewise, this release is identical to 0.15.1.

## [0.15.1] - Feb. 10, 2024

This release reaches milestone of 100% compliance of `dpctl.tensor` functions with Python Array API 2022.12 standard for the main namespace.

### Added

* Added reduction functions `dpctl.tensor.min`, `dpctl.tensor.max`, `dpctl.tensor.argmin`, `dpctl.tensor.argmax`, and `dpctl.tensor.prod` per Python Array API specifications: [#1399](https://github.com/IntelPython/dpctl/pull/1399)
* Added dedicated in-place operations for binary elementwise operations and deployed them in Python operators of `dpctl.tensor.usm_ndarray` type: [#1431](https://github.com/IntelPython/dpctl/pull/1431), [#1447](https://github.com/IntelPython/dpctl/pull/1447)
* Added new elementwise functions `dpctl.tensor.cbrt`, `dpctl.tensor.rsqrt`, `dpctl.tensor.exp2`, `dpctl.tensor.copysign`, `dpctl.tensor.angle`, and `dpctl.tensor.reciprocal`: [#1443](https://github.com/IntelPython/dpctl/pull/1443), [#1474](https://github.com/IntelPython/dpctl/pull/1474)
* Added statistical functions `dpctl.tensor.mean`, `dpctl.tensor.std`, `dpctl.tensor.var` per Python Array API specifications: [#1465](https://github.com/IntelPython/dpctl/pull/1465)
* Added sorting functions `dpctl.tensor.sort` and `dpctl.tensor.argsort`, and set functions `dpctl.tensor.unique_values`, `dpctl.tensor.unique_counts`, `dpctl.tensor.unique_inverse`, `dpctl.tensor.unique_all`: [#1483](https://github.com/IntelPython/dpctl/pull/1483)
* Added linear algebra functions from the Array API namespace `dpctl.tensor.matrix_transpose`, `dpctl.tensor.matmul`, `dpctl.tensor.vecdot`, and `dpctl.tensor.tensordot`: [#1490](https://github.com/IntelPython/dpctl/pull/1490), [#1525](https://github.com/IntelPython/dpctl/pull/1525), [#1541](https://github.com/IntelPython/dpctl/pull/1541)
* Added `dpctl.tensor.clip` function: [#1444](https://github.com/IntelPython/dpctl/pull/1444), [#1505](https://github.com/IntelPython/dpctl/pull/1505)
* Added custom reduction functions `dpt.logsumexp` (reduction using binary function `dpctl.tensor.logaddexp`), `dpt.reduce_hypot` (reduction using binary function `dpctl.tensor.hypot`): [#1446](https://github.com/IntelPython/dpctl/pull/1446)
* Added inspection API to query capabilities of Python Array API specification implementation: [#1469](https://github.com/IntelPython/dpctl/pull/1469)
* Support for compilation for NVIDIA(R) sycl target with use of [CodePlay oneAPI plug-in](https://developer.codeplay.com/products/oneapi/nvidia/home/): [#1411](https://github.com/IntelPython/dpctl/pull/1411), [#1124](https://github.com/IntelPython/dpctl/discussions/1124)
* Added `dpctl.utils.intel_device_info` function to query additional information about Intel(R) GPU devices: [gh-1428](https://github.com/IntelPython/dpctl/pull/1428) and [gh-1445](https://github.com/IntelPython/dpctl/pull/1445)
* Added support for two new device descriptors, `dpctl.SyclDevice.max_mem_alloc_size` and `dpctl.SyclDevice.max_clock_frequency`: [#1530](https://github.com/IntelPython/dpctl/pull/1530)

### Changed

* Functions `dpctl.tensor.result_type` and `dpctl.tensor.can_cast` became device-aware: [#1488](https://github.com/IntelPython/dpctl/pull/1488), [#1473](https://github.com/IntelPython/dpctl/pull/1473)
* Implementation of method `dpctl.SyclEvent.wait_for` changed to use ``sycl::event::wait`` instead of ``sycl::event::wait_and_throw``: [gh-1436](https://github.com/IntelPython/dpctl/pull/1436)
* `dpctl.tensor.astype` was changed to support `device` keyword as per Python Array API specification: [#1511](https://github.com/IntelPython/dpctl/pull/1511)
* C++ header files in `libtensor/include/kernels` containing implementations of SYCL kernels no longer depends on "pybind11.h": [#1516](https://github.com/IntelPython/dpctl/pull/1516)

### Fixed

* Fixed issues with `dpctl.tensor.repeat` support for `axis` keyword: [#1427](https://github.com/IntelPython/dpctl/pull/1427), [#1433](https://github.com/IntelPython/dpctl/pull/1433)
* Fix for gh-1503 for bug `usm_ndarray.__setitem__`: [#1504](https://github.com/IntelPython/dpctl/pull/1504)
* Other bug fixes: [#1485](https://github.com/IntelPython/dpctl/pull/1485), [#1477](https://github.com/IntelPython/dpctl/pull/1477), [#1512](https://github.com/IntelPython/dpctl/pull/1512)


## [0.15.0] - Sep. 29, 2023

### Added

* Added `dpctl.tensor.floor`, `dpctl.tensor.ceil`, `dpctl.tensor.trunc` elementwise functions.
* Added `dpctl.tensor.hypot`, `dpctl.tensor.logaddexp` elementwise functions.
* Added trigonometric (`dpctl.tensor.sin`, `dpctl.tensor.cos`, `dpctl.tensor.tan`) and hyperbolic (`dpctl.tensor.sinh`, `dpctl.tensor.cosh`, `dpctl.tensor.tanh`) elementwise functions and their inverses (`dpctl.tensor.asin`, `dpctl.tensor.asinh`, `dpctl.tensor.acos`, `dpctl.tensor.acosh`, `dpctl.tensor.atan`, `dpctl.tensor.atanh`).
* Added `dpctl.tensor.round` function.
* Added `dpctl.tensor.sign` and `dpctl.tensor.remainder` elementwise functions.
* Added bitwise elementwise functions `dpctl.tensor.bitwise_and`, `dpctl.tensor.bitwise_xor`, `dpctl.tensor.bitwise_or`, `dpctl.tensor.bitwise_invert`
* Added bitwise shift functions `dpctl.tensor.bitwise_left_shift` and `dpctl.tensor.bitwise_right_shift`.
* Added `dpctl.tensor.atan2` and `dpctl.tensor.signbit` elementwise functions.
* Added `dpctl.tensor.minumum` and `dpctl.tensor.maximum` binary elementwise functions.
* Supported equality checking and hashing for `dpctl.SyclPlatform`.
* Implemented `types` property for all unary and binary elementwise functions  [#1361](https://github.com/IntelPython/dpctl/pull/1361)
* Added `dpctl.tensor.repeat` and `dpctl.tensor.tile` functions.
* Added `dpctl.tensor.matrix_transpose ` function.

### Changed

* Enabled support for Python arithmetic, in-place arithmetic, reflexive arithmetic, comparison, and bitwise operators for `dpctl.tensor.usm_ndarray` type [#1324](https://github.com/IntelPython/dpctl/pull/1324).
* Removed `dpctl.tensor.numpy_usm_shared` obsolete class and associated tests which were being skipped [#1310](https://github.com/IntelPython/dpctl/pull/1310)
* Transitioned `dpctl` codebase to Cython 3.
* Improved performance of boolean reduction functions `dpctl.tensor.all` and `dpctl.tensor.any`.
* Improved performance of summation function `dpctl.tensor.sum`.
* Improved in-place arithmetic operations for addition, subtraction and multiplication.
* Updated codebase per SYCL-2020 intel/llvm compiler deprecation warnings.
* Improved performance of advanced boolean indexing for arrays whose size fits in 32-bit signed integer type.
* Removed deprecated `DPCTLDevice_GetMaxWorkItemSizes` function from the SyclInterface library.
* Improved performance of `dpctl.tensor.reshape` in the case when a copy is being made.
* Improved performance of `dpctl.tensor.roll` function.

### Fixed

* Fixed issues identified by Coverity security scans.
* Fixed issues [#1279](https://github.com/IntelPython/dpctl/issues/1279), [#1350](https://github.com/IntelPython/dpctl/issues/1350), [#1344](https://github.com/IntelPython/dpctl/issues/1344), [#1327](https://github.com/IntelPython/dpctl/issues/1327), [#1241](https://github.com/IntelPython/dpctl/issues/1241), [#1250](https://github.com/IntelPython/dpctl/issues/1250), [#1293](https://github.com/IntelPython/dpctl/issues/1293).

## [0.14.5] - 07/17/2023

### Added

* Added `dpctl.tensor.log2` and `dpctl.tensor.log10`: [#1267](https://github.com/IntelPython/dpctl/pull/1267)
* Added `dpctl.tensor.negative`, `dpctl.tensor.positive`, `dpctl.tensor.square` [#1268](https://github.com/IntelPython/dpctl/pull/1268)
* Added `dpctl.tensor.logical_not`, `dpctl.tensor.logical_and`, `dpctl.tensor.logical_or`, `dpctl.tensor.logical_xor` [#1270](https://github.com/IntelPython/dpctl/pull/1270)

### Changed

* `dpctl.tensor.astype` behavior for `newdtype=None` changes [#1261](https://github.com/IntelPython/dpctl/pull/1262)
* `dpctl.tensor.usm_ndaray` constructor default value of `dtype` keyword argument changed to `None`: [#1265](https://github.com/IntelPython/dpctl/pull/1265)
* Support for `out` arguments that overlap with inputs for unary elementwise functions[#1281](https://github.com/IntelPython/dpctl/pull/1281)
* Copying from one array to another a no-op if both arrays view into the same memory [#1284](https://github.com/IntelPython/dpctl/pull/1284)

## [0.14.4] - 06/14/2023

### Added

* Added `dpctl.tensor.less_equal`, `dpctl.tensor.greater`, `dpctl.tensor.greater_equal`: [#1239](https://github.com/IntelPython/dpctl/pull/1239)

### Changed

* Optimized in-place arithmetic operations for updating matrix with rows/columns via broadcasting: [#1244](https://github.com/IntelPython/dpctl/pull/1244)

### Fixed

* Fixed handling of 0d arrays in `dpctl.tensor.sum`: [#1238](https://github.com/IntelPython/dpctl/pull/1238)

## [0.14.3] - 06/13/2023

### Added

* Added support of `axis=None` in `dpctl.tensor.concat` [#1125](https://github.com/IntelPython/dpctl/pull/1125)
* Added caching for `dpctl.SyclDevice.filter_string` property [#1127](https://github.com/IntelPython/dpctl/pull/1127)
* Added `dpctl.tensor.isdtype` from array API [#1133](https://github.com/IntelPython/dpctl/pull/1133)
* Added `dpctl.tensor.unstack`, `dpctl.tensor.moveaxis`, `dpctl.tensor.swapaxes` [#1137](https://github.com/IntelPython/dpctl/pull/1137), [#1174](https://github.com/IntelPython/dpctl/pull/1174)
* Allow for mutation of `dpctl.tensor.usm_ndarray.flags.writable` [#1141](https://github.com/IntelPython/dpctl/pull/1141)
* Added `dpctl.tensor.where` from array API [#1147](https://github.com/IntelPython/dpctl/pull/1147)
* Include libtensor headers in `dpctl` installation layout [#1185](https://github.com/IntelPython/dpctl/pull/1185)
* Added new properties of `dpctl.tensor.usm_ndarray` object [#1199](https://github.com/IntelPython/dpctl/pull/1199)
* Added a list of unary and binary elementwise functions from array API:
   - [#1203](https://github.com/IntelPython/dpctl/pull/1203): `dpctl.tensor.add`, `dpctl.tensor.divide`, `dpctl.tensor.isnan`, `dpctl.tensor.isinf`, `dpctl.tensor.isfinite`, `dpctl.tensor.cos`, `dpctl.tensor.abs`, `dpctl.tensor.equal`
   - [#1205](https://github.com/IntelPython/dpctl/pull/1205): `dpctl.tensor.sqrt`
   - [#1209](https://github.com/IntelPython/dpctl/pull/1209): implements `out` keyword argument
   - [#1211](https://github.com/IntelPython/dpctl/pull/1211): `dpctl.tensor.multiply`, `dpctl.tensor.subtract`
   - [#1214](https://github.com/IntelPython/dpctl/pull/1214): `dpctl.tensor.not_equal`
   - [#1216](https://github.com/IntelPython/dpctl/pull/1216): `dpctl.tensor.exp`, `dpctl.tensor.sin`
   - [#1217](https://github.com/IntelPython/dpctl/pull/1217): `dpctl.tensor.real`, `dpctl.tensor.imag`, `dpctl.tensor.proj`
   - [#1218](https://github.com/IntelPython/dpctl/pull/1218): `dpctl.tensor.log`, `dpctl.tensor.log1p`, `dpctl.tensor.expm1`
   - [#1221](https://github.com/IntelPython/dpctl/pull/1221): `dpctl.tensor.floor_divide`
   - [#1235](https://github.com/IntelPython/dpctl/pull/1235): `dpctl.tensor.less`
   - [#1237](https://github.com/IntelPython/dpctl/pull/1237): in-place support for addition, multiplication and subtraction
* Added `dpctl.tensor.all` and `dpctl.tensor.any` [#1204](https://github.com/IntelPython/dpctl/pull/1204)
* Added `dpctl.tensor.sum` [#1210](https://github.com/IntelPython/dpctl/pull/1210)

### Changed

* Updated examples of native Python extensions built using `dpctl` [#1108](https://github.com/IntelPython/dpctl/pull/1108)
* Used security flags to compile and link native extensions of `dpctl` [#1109](https://github.com/IntelPython/dpctl/pull/1109)
* Changed types of `dpctl.tensor.finfo` and `dpctl.tensor.iinfo` output structure per array API spec [#1110](https://github.com/IntelPython/dpctl/pull/1110)
* Consolidated multiple USM temporaries life-time management `host_task`s to improve test suite stability [#1111](https://github.com/IntelPython/dpctl/pull/1111)
* MAINT: Improved cmake target dependency tracking [#1112](https://github.com/IntelPython/dpctl/pull/1112)
* MAINT: Improved docstrings for existing `dpctl.tensor` functions [#1123](https://github.com/IntelPython/dpctl/pull/1123)
* Changed default value of `mode` keyword in `dpctl.tensor.take` and `dpctl.take.put` from `clip` to `wrap` [#1132](https://github.com/IntelPython/dpctl/pull/1132)
* Added support for (nested) sequence of `dpctl.tensor.usm_ndarray` objects in `dpctl.tensor.asarray` [#1139](https://github.com/IntelPython/dpctl/pull/1139)
* Improved exception handling in `dpctl.tensor.usm_ndarray.__setitem__` special method [#1146](https://github.com/IntelPython/dpctl/pull/1146)
* Simplified implementation of copy-and-cast kernels and removed special casing for 2D arrays to conserve binary size [#1165](https://github.com/IntelPython/dpctl/pull/1165)
* Improved speed of `dpctl.tensor.usm_ndarray` printing functionality [#1187](https://github.com/IntelPython/dpctl/pull/1187)
* Require DPC++ RT 2023.1 to build and run `dpctl` [#1195](ttps://github.com/IntelPython/dpctl/pull/1195)
* Compile offloading native extensions with `-fno-sycl-id-queries-fit-in-int` fixing [gh-1184](https://github.com/IntelPython/dpctl/issues/1184), [#1200](https://github.com/IntelPython/dpctl/pull/1200)
* Transition to conda-forge ecosystem [#1213](https://github.com/IntelPython/dpctl/pull/1213)



### Fixed

* Fix to add empty values check for `dpctl.tensor.place` [#1105](https://github.com/IntelPython/dpctl/pull/1105), [#1106](https://github.com/IntelPython/dpctl/pull/1106)
* Fixed [gh-1089](https://github.com/IntelPython/dpctl/issues/1089) by improving `dpctl.tensor.asarray` handling of NumPy arrays viewing into host-accessible USM allocation objects.
* MAINT: Fixed build break with newer GCC and SYCLOS [#1118](https://github.com/IntelPython/dpctl/pull/1118)
* Fixed a bug in basic indexing of `dpctl.tensor.usm_ndarray` [#1136](https://github.com/IntelPython/dpctl/pull/1136)




## [0.14.2] - 03/07/2023

### Fixed

* Fixed a bug with boolean advanced indexing [#1103](https://github.com/IntelPython/dpctl/pull/1103)

## [0.14.1] - 03/06/2023

### Added

* Added `dpctl.SyclDevice.partition_max_sub_devices` property [#1005](https://github.com/IntelPython/dpctl/pull/1005)
* Added `dpctl.program.SyclKernel.max_sub_group_size` property [#1028](https://github.com/IntelPython/dpctl/pull/1028)
* Implemented printing of `usm_ndarray` [#1013](https://github.com/IntelPython/dpctl/pull/1013), [#1043](https://github.com/IntelPython/dpctl/pull/1043), [#1060](https://github.com/IntelPython/dpctl/pull/1060)
* Implemented support for advanced indexing for `dpctl.tensor.usm_ndarray` [#1095](https://github.com/IntelPython/dpctl/pull/1095), [#1097](https://github.com/IntelPython/dpctl/pull/1097), [#1099](https://github.com/IntelPython/dpctl/pull/1099), [#1101](https://github.com/IntelPython/dpctl/pull/1101)
* Implemented support for platform listing in `dpctl.__main__` script [#1014](https://github.com/IntelPython/dpctl/pull/1014)
* Improved performance of `dpctl.tensor.asnumpy` [#1026](https://github.com/IntelPython/dpctl/pull/1026)
* Added `UsmNDArray_Make*` C-API for constructing `dpctl.tensor.usm_ndarray` from native allocations [#1050](https://github.com/IntelPython/dpctl/pull/1050), [#1067](https://github.com/IntelPython/dpctl/pull/1067)
* Added support for `dpctl.SyclDevice.native_vector_width_*` device descriptors [#1075](https://github.com/IntelPython/dpctl/pull/1075)
* Added ``dpctl::tensor::usm_ndarray::get_shape_vector`` and ``dpctl::tensor::usm_ndarray::get_strides_vector`` methods [#1090](https://github.com/IntelPython/dpctl/pull/1090)


### Changed

* Removed `dpctl.select_host_device`, `dpctl.has_host_device`, `dpctl.SyclDevice.is_host`, and `dpctl.SyclDevice.has_aspect_host` since support for host device has been removed in DPC++ 2023 and from SYCL 2020 spec [#1028](https://github.com/IntelPython/dpctl/pull/1028)
* `usm_ndarray`is made writable by default [#1012](https://github.com/IntelPython/dpctl/pull/1012), and writable flag is now checked by `__setitem__`.
* Added convenience signature for C++ utility function in "dpctl4pybind11.hpp" [#1016](https://github.com/IntelPython/dpctl/pull/1016)
* Improved error reported when attempting to submit kernel that uses a data-type unsupported by target device [#1018](https://github.com/IntelPython/dpctl/pull/1018), [#1040](https://github.com/IntelPython/dpctl/pull/1040)
* Updated C++ code to require DPC++ 2023.0.0 or newer [#1028](https://github.com/IntelPython/dpctl/pull/1028), [#1066](https://github.com/IntelPython/dpctl/pull/1066)
* The `dpctl.tensor.Device` class supports `print_device_info` method [#1029](https://github.com/IntelPython/dpctl/pull/1029), equality comparison, and hashing [#1048](https://github.com/IntelPython/dpctl/pull/1048)
* Updated version of pybind11 used to 2.10.2 [#1031](https://github.com/IntelPython/dpctl/pull/1031)
* Improved internal utility responsible for reduction of iteration space dimensionality [#1044](https://github.com/IntelPython/dpctl/pull/1044/), [#1054](https://github.com/IntelPython/dpctl/pull/1054)
* Changed return type of `DCPCTLUSM_GetPointerType` function in SyclInterface library [#1061](https://github.com/IntelPython/dpctl/pull/1061), [#1065](https://github.com/IntelPython/dpctl/pull/1065)
* Updated supported version of DLPack to 0.8 [#1073](https://github.com/IntelPython/dpctl/pull/1073)
* Implemented queue cache per context/device pair and deployed it in `dpctl.memory`, `dpctl.tensor.from_dlpack` and `dpctl.tensor` array creation functions [#1076](https://github.com/IntelPython/dpctl/pull/1076), [#1079](https://github.com/IntelPython/dpctl/pull/1079)

* Maintainance, CI work: [#1001](https://github.com/IntelPython/dpctl/pull/1001), [#1009](https://github.com/IntelPython/dpctl/pull/1009), [#1011](https://github.com/IntelPython/dpctl/pull/1011), [#1024](https://github.com/IntelPython/dpctl/pull/1024), [#1030](https://github.com/IntelPython/dpctl/pull/1030), [#1032](https://github.com/IntelPython/dpctl/pull/1032), [#1035](https://github.com/IntelPython/dpctl/pull/1035), [#1037](https://github.com/IntelPython/dpctl/pull/1037), [#1039](https://github.com/IntelPython/dpctl/pull/1039), [#1041](https://github.com/IntelPython/dpctl/pull/1041), [#1045](https://github.com/IntelPython/dpctl/pull/1045), [#1047](https://github.com/IntelPython/dpctl/pull/1047), [#1055](https://github.com/IntelPython/dpctl/pull/1055), [#1057](https://github.com/IntelPython/dpctl/pull/1057), [#1059](https://github.com/IntelPython/dpctl/pull/1059), [#1068](https://github.com/IntelPython/dpctl/pull/1068), [#1070](https://github.com/IntelPython/dpctl/pull/1070), [#1074](https://github.com/IntelPython/dpctl/pull/1074), [#1077](https://github.com/IntelPython/dpctl/pull/1077), [#1078](https://github.com/IntelPython/dpctl/pull/1078), [#1081](https://github.com/IntelPython/dpctl/pull/1081), [#1084](https://github.com/IntelPython/dpctl/pull/1084), [#1085](https://github.com/IntelPython/dpctl/pull/1085), [#1088](https://github.com/IntelPython/dpctl/pull/1088), [#1086](https://github.com/IntelPython/dpctl/pull/1086), [#1092](https://github.com/IntelPython/dpctl/pull/1092), [#1093](https://github.com/IntelPython/dpctl/pull/1093)


### Fixed

* Fixed error [gh-998](https://github.com/IntelPython/dpctl/issues/998) in forming Python exception, [#999](https://github.com/IntelPython/dpctl/pull/999).
* A small memory leak fixed, [#1000](https://github.com/IntelPython/dpctl/pull/1000)
* Improved dtype support in `dpctl.tensor.full`,  PR [#1002](https://github.com/IntelPython/dpctl/pull/1002)
* Added missing header file [#1008](https://github.com/IntelPython/dpctl/pull/1008) fixing [gh-1007](https://github.com/IntelPython/dpctl/issues/1007)
* Fixed a typo in device-specific dtype mapping [#1015](https://github.com/IntelPython/dpctl/pull/1015)
* Fixed default device integer type to align with NumPy's behavior on Windows [#1017](https://github.com/IntelPython/dpctl/pull/1017)
* Fixed unexpected overflow in `dpctl.tensor.linspace` when one of the parameters is the largest floating point value [#1034](https://github.com/IntelPython/dpctl/pull/1034)
* Constructors `dpctl.tensor.empty`, `dpctl.tensor.zeros`, and `usm_ndarray` constructor itself no longer allow to create array with data-types not supported by targeted device [#1042](https://github.com/IntelPython/dpctl/pull/1042)
* Fixed parameter validation in `dpctl.SyclQueue` constructor [#1052](https://github.com/IntelPython/dpctl/pull/1052)
* Fixed `usm_type` of the resulting array in `dpctl.tensor.tril` and `dpctl.tensor.triu` functions [#1062](https://github.com/IntelPython/dpctl/pull/1062)
* Used DPC++ configuration files to ensure correct use of conda compiler toolchain on Linux [#1072](https://github.com/IntelPython/dpctl/pull/1072)
* Fixed issue with empty argument of `dpctl.tensor.meshgrid` function [#1080](https://github.com/IntelPython/dpctl/pull/1080/)
* Fixed linking problem on Windows enabling `dpctl` to be functional on Windows for devices not supporting some data types [#1083](https://github.com/IntelPython/dpctl/pull/1083)

## [0.14.0] - 11/18/2022

### Added

* Implemented `dpctl.tensor.linspace` function from array-API [#875](https://github.com/IntelPython/dpctl/pull/875).
* Implemented `dpctl.tensor.eye` function from array-API [#896](https://github.com/IntelPython/dpctl/pull/896).
* Implemented `dpctl.tensor.tril` and `dpctl.tensor.triu` functions from array-API [#910](https://github.com/IntelPython/dpctl/pull/910).
* Added data type objects to `dpctl.tensor` namespace, `finfo`, `iinfo`, `can_cast`, and `result_type` functions [#913](https://github.com/IntelPython/dpctl/pull/913).
* Implemented `dpctl.tensor.meshgrid` creation function from array-API [#920](https://github.com/IntelPython/dpctl/pull/920).
* Implemented convenience class to represent output of `dpctl.tensor.usm_ndarray.flags` property [#921](https://github.com/IntelPython/dpctl/pull/921).
* Added new device attributes and kernel's device-specific attributes [#894](https://github.com/IntelPython/dpctl/pull/894).
* Added `dpctl.utils.onetrace_enabled` context manager for targeted trace collection [#903](https://github.com/IntelPython/dpctl/pull/903).
* Added support for `stream` keyword in `__dlpack__` method, enabling support for sending `usm_ndarray` using mpi4py [#906](https://github.com/IntelPython/dpctl/pull/906).
* `dpctl.tensor.asarray` can now transition data between incompatible devices, [#951](https://github.com/IntelPython/dpctl/pull/951).
* Introduced `"syclinterface/dpctl_sycl_types_casters.hpp"` header file with declaration of conversion routines between SYCL type pointers and SyclInterface library opaque pointers [#960](https://github.com/IntelPython/dpctl/pull/960).
* Added C-API to `dpctl.program.SyclKernel` and `dpctl.program.SyclProgram`. Added type casters for new types to "dpctl4pybind11" and added an example demonstrating its use [#970](https://github.com/IntelPython/dpctl/pull/970).
* Introduced "dpctl/sycl.pxd" Cython declaration file to streamline use of SYCL functions from Cython, and added an example demonstrating its use [#981](https://github.com/IntelPython/dpctl/pull/981).
* Added experimental support for sharing data allocated on sub-devices via dlpack [#984](https://github.com/IntelPython/dpctl/pull/984).
* Added `dpctl.SyclDevice.sub_group_sizes` property to retrieve supported sizes of sub-group by the device [#985](https://github.com/IntelPython/dpctl/pull/985).

### Changed
* Improved queue compatibility testing in `dpctl.tensor`'s implementation module [#900](https://github.com/IntelPython/dpctl/pull/900).
* Added automatic measurement of array-API conformance test suite in CI [#901](https://github.com/IntelPython/dpctl/pull/901).
* Improved performance of array metadata transfer from host to device [#912](https://github.com/IntelPython/dpctl/pull/912).
* Used `os.add_dll_directory` on Windows to ensure that `DPCTLSyclInterface` library can be found [#918](https://github.com/IntelPython/dpctl/pull/918).
* Refactored `dpctl.tensor`'s implementation module [#941](https://github.com/IntelPython/dpctl/pull/941) to streamline adding new functionality. Streamlined `dpctl::tensor::usm_ndarray` class implementation.
* Added debugging messaging in case when `DPCTLDynamicLib::getSymbol` encounters errors [#956](https://github.com/IntelPython/dpctl/pull/956).
* Updated code base according to changes in DPC++ compiler [#952](https://github.com/IntelPython/dpctl/pull/952), [#957](https://github.com/IntelPython/dpctl/pull/957), [#958](https://github.com/IntelPython/dpctl/pull/958).
* Changed `dpctl` to use pybind11 2.10.1 [#967](https://github.com/IntelPython/dpctl/pull/967).
* Extended `dpctl.tensor.full` to accept 0d and higher dimensional arrays for fill-value parameter [#982](https://github.com/IntelPython/dpctl/pull/982) and [#995](https://github.com/IntelPython/dpctl/pull/995).

### Fixed
* Improved SyclDevice constructor error message [#893](https://github.com/IntelPython/dpctl/pull/893).
* Fixed issue gh-890 about `dpctl.tensor.reshape` function [#915](https://github.com/IntelPython/dpctl/pull/915).
* Fixed unexpected `UnboundLocalError` exception in [#922](https://github.com/IntelPython/dpctl/pull/922).
* Fixed bugs in `dpctl.tensor.arange` in [#945](https://github.com/IntelPython/dpctl/pull/945).
* Fixed issue with type inferencing in `dpctl.tensor.asarray` in [#949](https://github.com/IntelPython/dpctl/pull/949).
* Added missing docstrings for `dpctl.SyclDevice` properties [#964](https://github.com/IntelPython/dpctl/pull/964).

## [0.13.0] - 07/28/2022

### Added

* Implemented and deployed dedicated kernels for copying with casting [#781](https://github.com/IntelPython/dpctl/781), used in `__setitem__`, implementaion of `asarray`, `dpctl.tensor.copy` functions.
* Implemented dedicated copying kernel for `dpctl.tensor.reshape` function [#810](https://github.com/IntelPython/dpctl/810), added support for `copy` keyword [#807](https://github.com/IntelPython/dpctl/807).
* Implemented dedicated kernel to copy with casting from `numpy.ndarray` into `dpctl.tensor.usm_ndarray` [#817](https://github.com/IntelPython/dpctl/pull/817).

* Implemented `dpctl.tensor.permute_dims` function from array-API [#787](https://github.com/IntelPython/dpctl/pull/787).
* Implemented `dpctl.tensor.expand_dims` function from array-API [#788](https://github.com/IntelPython/dpctl/pull/788).
* Implemented `dpctl.tensor.squeeze` function from array-API [#790](https://github.com/IntelPython/dpctl/pull/790).
* Implemented `dpctl.tensor.broadcast_to` function from array-API [#791](https://github.com/IntelPython/dpctl/791).
* Implemented `dpctl.tensor.broadcast_arrays` function from array-API [#798](https://github.com/IntelPython/dpctl/798).
* Implemented `dpctl.tensor.flip` function from array-API [#801](https://github.com/IntelPython/dpctl/801).
* Implemented `dpctl.tensor.usm_ndarray.mT` property per array-API [#805](https://github.com/IntelPython/dpctl/805).
* Implemented `dpctl.tensor.roll` function from array-API [#809](https://github.com/IntelPython/dpctl/809).
* Implemented `dpctl.tensor.arange` function from array-API [#814](https://github.com/IntelPython/dpctl/814).
* Implemented `dpctl.tensor.zeros` function from array-API [#816](https://github.com/IntelPython/dpctl/816).
* Implemented `dpctl.tensor.zeros` function from array-API [#816](https://github.com/IntelPython/dpctl/816).
* Implemented `dpctl.tensor.ones`, `dpctl.tensor.full`, `dpctl.tensor.empty_like`, `dpctl.tensor.zeros_like`, `dpctl.tensor.ones_like`, `dpctl.tensor.full_like` functions from array-API [#822](https://github.com/IntelPython/dpctl/pull/822).
* Implemented `DPCTLQueue_Memset` function in SyclInterface library [#812](https://github.com/IntelPython/dpctl/812), and exposed it for `dpctl.memory.MemoryUSM*` classes [#815](https://github.com/IntelPython/dpctl/815).
* Implemented `dpctl.utils.get_coerced_usm_type` to deduced usm type of the output array from types of input arrays in compute-follows-data execution model [#797](https://github.com/IntelPython/dpctl/pull/797).
* Added `dpctl.SyclDevice.profiling_timer_resolution` property [#825](https://github.com/IntelPython/dpctl/pull/825).
* Added `dpctl.SyclDevice.platform` and `dpctl.SyclPlatform.default_context` properties [#827](https://github.com/IntelPython/dpctl/pull/827).
* Provided pybind11 example for functions working on `dpctl.tensor.usm_ndarray` container applying oneMKL functions [#780](https://github.com/IntelPython/dpctl/pull/780), [#793](https://github.com/IntelPython/dpctl/pull/793), [#819](https://github.com/IntelPython/dpctl/pull/819). The example was expanded to demonstrate implementing iterative linear solvers (Chebyshev solver, and Conjugate-Gradient solver) by asynchronously submitting individual SYCL kernels from Python [#821](https://github.com/IntelPython/dpctl/pull/821), [#833](https://github.com/IntelPython/dpctl/pull/833), [#838](https://github.com/IntelPython/dpctl/pull/838).
* Wrote manual page about working with `dpctl.SyclQueue` [#829](https://github.com/IntelPython/dpctl/pull/829).
* Added cmake scripts to dpctl package layout and a way to query the location [#853](https://github.com/IntelPython/dpctl/pull/853).
* Implemented `dpctl.tensor.concat` function from array-API [#867](https://github.com/IntelPython/dpctl/867).
* Implemented `dpctl.tensor.stack` function from array-API [#872](https://github.com/IntelPython/dpctl/872).


### Changed

* Enhanced coverage collection for SyclInterface library by also collecting it during pytest run and combining traces with those collected during C-test run [#818](https://github.com/IntelPython/dpctl/pull/818). This change also allows to not rebuild SyclInterface library when building C-test executable.
* Exported `keep_args_alive` utility in `dpctl4pybind11.hpp` header [#820](https://github.com/IntelPython/dpctl/pull/820). The utility uses `sycl::handler::host_task` to keep given Python arguments alive until eac `sycl::event` from the given vector of events is complete. The host task is scheduled on the SYCL queue provided as the first argument.
* Changed the size of struct underlying `dpctl.SyclEvent` to avoid storing Python object previously used to keep kernel arguments scheduled with `dpctl.SyclQueue.submit` [#823](https://github.com/IntelPython/dpctl/pull/823).
* Fixed docstring for `dpctl.SyclTimer` [#824](https://github.com/IntelPython/dpctl/pull/824).
* Changed type of exceptions raised on failure to create `dpctl.SyclDevice` from `ValueError` to `dpctl.SyclDeviceCreationError` [#826](https://github.com/IntelPython/dpctl/pull/826).
* Improved performance of pybind11 type casters [#837](https://github.com/IntelPython/dpctl/pull/837).
* Changed implementation of `dpctl.SyclProgram` from using deprecated `sycl::program` to `sycl::kernel_bundle` [#845](https://github.com/IntelPython/dpctl/pull/845).
* Removed deprecated device aspects, added new supported aspects [#844](https://github.com/IntelPython/dpctl/pull/844).
* Updated vendored `dlpack.h` to version 0.7 [#847](https://github.com/IntelPython/dpctl/pull/847).

### Fixed

* Fixed `dpctl.lsplatform()` to work correctly when used from within Jupyter notebook [#800](https://github.com/IntelPython/dpctl/pull/800).
* Fixed script to drive debug build [#835](https://github.com/IntelPython/dpctl/pull/835) and fixed code to compile in debug mode [#836](https://github.com/IntelPython/dpctl/pull/836).
* Fixed filter selector string produced in outputs of `dpctl.lsplatform(verbosity=2)` and `dpctl.SyclDevice.print_device_info` [#866](https://github.com/IntelPython/dpctl/pull/866).
* Fixed issue with slicing reported in gh-870 in [#871](https://github.com/IntelPython/dpctl/pull/871).

## [0.12.0] - 03/01/2022

### Added

* Properties added to MemoryUSM* objects. [#647](https://github.com/IntelPython/dpctl/pull/647)
* Added `dpctl.tensor.asarray` [#646](https://github.com/IntelPython/dpctl/pull/646)
* Implemented DLPack support for usm_ndarray [#682](https://github.com/IntelPython/dpctl/pull/682)
* Exported `dpctl.tensor.Device` class [#708](https://github.com/IntelPython/dpctl/pull/708) [#718](https://github.com/IntelPython/dpctl/pull/718)
* Added testing of examples in CI [#722](https://github.com/IntelPython/dpctl/pull/722)
* Added user manuals to dpctl documentation [#712](https://github.com/IntelPython/dpctl/pull/712) [#773](https://github.com/IntelPython/dpctl/pull/773)

### Changed

* Folder dpctl-capi/ renamed to libsyclinterface/ in sources and documentation.  [#666](https://github.com/IntelPython/dpctl/pull/666)
 [#768](https://github.com/IntelPython/dpctl/pull/768)
* Added workflow to publish rendered documentation on PRs [#673](https://github.com/IntelPython/dpctl/pull/673) [#753](https://github.com/IntelPython/dpctl/pull/753) [#726](https://github.com/IntelPython/dpctl/pull/726)
* Synchronization functions and USM allocation functions release GIL [#736](https://github.com/IntelPython/dpctl/pull/736) [#766](https://github.com/IntelPython/dpctl/pull/766)
* `dpctl.SyclEvent` destructor is made non-blocking [#751](https://github.com/IntelPython/dpctl/pull/751)

### Fixed
* Fixed for issue in code of `dpctl.tensor.usm_ndarray.T` [#653](https://github.com/IntelPython/dpctl/pull/653)
* Fixed issue with `dpctl.tensor.reshape`'s affect on contiguity flags of usm_ndarray [#695](https://github.com/IntelPython/dpctl/pull/695)
* Fixed handling of empty list by `dpctl.tensor.asarray` [#694](https://github.com/IntelPython/dpctl/pull/694)
* Fixed type inference with array of empty arrays in `dpctl.tensor.asarray` [#697](https://github.com/IntelPython/dpctl/pull/697)
* Fixed issue gh-698 with `dpctl.tensr.asarray` [#709](https://github.com/IntelPython/dpctl/pull/709)
* Fixed performance of item assignment from numpy array [#724](https://github.com/IntelPython/dpctl/pull/724)
* `DPCTLDeviceMgr_GetNumDevices` should not operate on rejected devices [#737](https://github.com/IntelPython/dpctl/pull/737)
* Fixed issue gh-729 for `dpctl.tensor.reshape` applied to 0-element usm_ndarray [#756](https://github.com/IntelPython/dpctl/pull/756)
* Fixed issue gh-728 with `dpctl.tensor.astype` [#757](https://github.com/IntelPython/dpctl/pull/757)
* Fixed type in memory overlapping test [#770](https://github.com/IntelPython/dpctl/pull/770)
* Fixed issue with operator.pos for `dpctl.tensor.usm_ndarray` [#783](https://github.com/IntelPython/dpctl/pull/783)
* Only call `PyThread_Ensure` from host_task if the main-thread interpreter is initialized and not finalizing [#776](https://github.com/IntelPython/dpctl/pull/776) [#778](https://github.com/IntelPython/dpctl/pull/778) [#721](https://github.com/IntelPython/dpctl/pull/721)

**Full Changelog**: https://github.com/IntelPython/dpctl/compare/0.11.4...0.12.0

## [0.11.4] - 12/03/2021

### Fixed
- Fix tests for nested context factories expecting for integration environment by @PokhodenkoSA in https://github.com/IntelPython/dpctl/pull/705

## [0.11.3] - 11/30/2021

### Fixed
- Set the last byte in allocated char array to zero [cherry picked from #650] [#699](https://github.com/IntelPython/dpctl/pull/699)

## [0.11.2] - 11/29/2021

### Added
- Extending `dpctl.device_context` with nested contexts [#678](https://github.com/IntelPython/dpctl/pull/678)

### Fixed
- Fixed issue #649 about incorrect behavior of `.T` method on sliced arrays [#653](https://github.com/IntelPython/dpctl/pull/653)

## [0.11.1] - 11/10/2021

### Changed
- Replaced uses of clang compiler with icx executable [#665](https://github.com/IntelPython/dpctl/pull/665)

## [0.11.0] - 11/01/2021

### Added
- Use Python 3.9 in public CI [#599](https://github.com/IntelPython/dpctl/pull/599)
- Add a new C API utility function (`DPCTLDeviceMgr_GetDeviceInfoStr`) to return the device info as a C string object [#620](https://github.com/IntelPython/dpctl/pull/620)
- New Github workflow to build dpclt with nightly Intel llvm/sycl + drivers [#621](https://github.com/IntelPython/dpctl/pull/621)
- Always raise SubDeviceCreationError even when sub-device counts are zero [#622](https://github.com/IntelPython/dpctl/pull/622)
- Updated OpenCL interoprability code to fix build with Intel llvm/sycl bundle [#625](https://github.com/IntelPython/dpctl/pull/625)
- Enabled use of default platform context extension in SYCL compilers that implement this extension [#627](https://github.com/IntelPython/dpctl/pull/627)
- Implemented `dpctl.utils.get_execution_queue(queue_seq)` utility to help implementing "compute-follows data" convention for offload target [#632](https://github.com/IntelPython/dpctl/pull/632)
 [#631](https://github.com/IntelPython/dpctl/pull/631)

### Changed
- Replaced `host_device` device type with `host` in tests [#616](https://github.com/IntelPython/dpctl/pull/616)
- Rework the logic in `dpctl.memory`'s `copy_from_device` method to work correctly with `host` device [#618](https://github.com/IntelPython/dpctl/pull/618)
- Use `dpctl.device_type.host` instead of `dpctl.device_type.host_device` [#626](https://github.com/IntelPython/dpctl/pull/626)
- Reinstate deprecated `sycl::program` and that was conditionally removed from open source DPC++ toolchain [#633](https://github.com/IntelPython/dpctl/pull/633)
- Use `LoadLibraryExA` instead of `LoadLibraryA` to mitigate a possible DLL injection issue when we load the Level zero DLL on windows [#636](https://github.com/IntelPython/dpctl/pull/636)
- Github coverage workflow is changed to use oneAPI 2021.3 instead of latest to work around broken profiling instrumentation in DPC++ 2021.4 [#614](https://github.com/IntelPython/dpctl/pull/614)
- Update build dependencies for NumPy [#641](https://github.com/IntelPython/dpctl/pull/641)
- Use "readelf" on SYCL's `pi_level_zero` library to find out and use the exact name of `ze_loader.so` in SyclInterface library [#617](https://github.com/IntelPython/dpctl/pull/617)

### Removed
- Removed use of DPC++ features deprecated in 2021.4 and open source Intel llvm/sycl compiler [#603](https://github.com/IntelPython/dpctl/pull/603)

### Fixed
- Suppress errant CMake log [#610](https://github.com/IntelPython/dpctl/pull/610)
- Fixes to compile dpctl using Intel llvm/sycl compiler [#603](https://github.com/IntelPython/dpctl/pull/603)
- Fix for the hang is to avoid passing `nullptr` argument to `sycl::queue::prefetch` [#612](https://github.com/IntelPython/dpctl/pull/612)
- Fixed the logic to return device count [#623](https://github.com/IntelPython/dpctl/pull/623)
- Enabled building of C extensions with dpctl by including header defining `bool` type for C compilers [#604](https://github.com/IntelPython/dpctl/pull/604)

## [0.10.0] - 09/28/2021

### Added
- Added methods __bool__, __float__, __int__, __index__,
and __complex__ to usm_ndarray [#578](https://github.com/IntelPython/dpctl/pull/578)
- Added data-API required special methods to usm_ndarray class,
as well as to_numpy/from_numpy, astype, reshape functions [#586](https://github.com/IntelPython/dpctl/pull/586)
- Added methods to query dpctl.SyclDevice for size of global/local memory [#589](https://github.com/IntelPython/dpctl/pull/589)
- Added tests for constructors with invalid capsules [#577](https://github.com/IntelPython/dpctl/pull/577)
- Improved test coverage of `dpctl.SyclQueue` implementation [#574](https://github.com/IntelPython/dpctl/pull/574)
- Added a test to exercise API exported function (get_event_ref). [#570](https://github.com/IntelPython/dpctl/pull/570)
- Expanded tests in test_sycl_context to improve coverage [#571](https://github.com/IntelPython/dpctl/pull/571)
- Tweaks to test_sycl_event to improve coverage [#567](https://github.com/IntelPython/dpctl/pull/567)
- Improved coverage of dpctl.__init__ file and other service functions [#563](https://github.com/IntelPython/dpctl/pull/563)
- Added test for repr and test for default argument to constructor [#565](https://github.com/IntelPython/dpctl/pull/565)
- Added some tests to involve capsule [#564](https://github.com/IntelPython/dpctl/pull/564)
- Added workflow for Public CI on Windows [#534](https://github.com/IntelPython/dpctl/pull/534)
- DPCTLQueue_Memcpy, _Prefetch, _Memadvise become asynchronous [#557](https://github.com/IntelPython/dpctl/pull/557)
- Added device aspect selector, `dpctl.select_device_with_aspects` [#558](https://github.com/IntelPython/dpctl/pull/558)
- Added test based on example from #583

### Changed
- Parametrized tests for executing OpenCL kernels compiled from source in types of arguments [#581](https://github.com/IntelPython/dpctl/pull/581)
- Temporary disabled self-hosted CI jobs runner [#559](https://github.com/IntelPython/dpctl/pull/559)
- Changed static method `SyclQueue._create_from_context_and_device` [#579](https://github.com/IntelPython/dpctl/pull/579)
- Transitioned all Python API to use pytest over unittest, improved coverage in dpctl/memory [#575](https://github.com/IntelPython/dpctl/pull/575)
- Changed `dpctl.SyclEvent.profiling_info_submit` from method to a property [#573](https://github.com/IntelPython/dpctl/pull/573)
- Simplified arg parsing in SyclDevice constructor [#572](https://github.com/IntelPython/dpctl/pull/572)
- Used<img> tag with alignment attribute set in README [#562](https://github.com/IntelPython/dpctl/pull/562)
- Moved sycl timer into dpctl.SyclTimer [#555](https://github.com/IntelPython/dpctl/pull/555)
- Used clang-format off, clang-format on to avoid include reordering in pybind11 example [#588](https://github.com/IntelPython/dpctl/pull/588)

### Fixed
- Implemented a workaround for running conda-build using Klocwork [#566](https://github.com/IntelPython/dpctl/pull/566)
- Separated pipelines for Linux and Windows [#582](https://github.com/IntelPython/dpctl/pull/582)
- Fixed inconsistency in `__sycl_usm_array_interface__` of `usm_ndarray` instance [#584](https://github.com/IntelPython/dpctl/pull/584)
- Fixed memory leak: Capsule deleters now free resources for renamed capsules too [#568](https://github.com/IntelPython/dpctl/pull/568)
- Fixed __version__ test to allow for semantic versioning [#569](https://github.com/IntelPython/dpctl/pull/569)
- Improved coverage of _types.pxi [#556](https://github.com/IntelPython/dpctl/pull/556)
- Fixed `UnboundLocalError` when default queue could not be created [#554](https://github.com/IntelPython/dpctl/pull/554)

## [0.9.0] - 08/25/2021

### Added
- Improvements to logic for working with custom DPC++ toolchain [#481](https://github.com/IntelPython/dpctl/pull/481)
- Add SyclContext unit test cases [#488](https://github.com/IntelPython/dpctl/pull/488)
- Consolidate configurations of tools that support PEP 518 into pyproject.toml [#486](https://github.com/IntelPython/dpctl/pull/486)
- Added C-API hash function, used them in Python interface [#491](https://github.com/IntelPython/dpctl/pull/491)
- Add missing extra checks to ensure unwrapped pointer is not Null
- Add error messages to L0 program creation routine
- Improve test coverage for dpctl_sycl_queue_interface [#492](https://github.com/IntelPython/dpctl/pull/492)
- Use pytest.warns in test_lsplatform3 [#495](https://github.com/IntelPython/dpctl/pull/495)
- Added test class to test DRef=nullptr case [#496](https://github.com/IntelPython/dpctl/pull/496)
- Extend parameterized test in test_sycl_queue_interface [#497](https://github.com/IntelPython/dpctl/pull/497)
- Use Memcpy, memadvise in tests
- Expanded types tests by TestQueueSubmitRange
- Added a test that retrieved DPCPP compiled kernel and submits them via DPCTLQueue_SubmitRange [#499](https://github.com/IntelPython/dpctl/pull/499)
, DPCTLEvent_GetCommandExecutionStatus [#516](https://github.com/IntelPython/dpctl/pull/516),
, DPCTLEvent_GetWaitList [#510](https://github.com/IntelPython/dpctl/pull/510) functions
- Propagate compile flags [#512](https://github.com/IntelPython/dpctl/pull/512)
- Add conda package CI pipeline on GitHub Actions [#515](https://github.com/IntelPython/dpctl/pull/515)
- Run tests on GPU [#518](https://github.com/IntelPython/dpctl/pull/518)
- Add 3 wrapper func for event::get_profiling_info [#519](https://github.com/IntelPython/dpctl/pull/519)
- Changes to build_backend.py to enable sycl-compiler-prefix on Windows
- dtype keyword of usm_ndarray now supports np.double and other types [#526](https://github.com/IntelPython/dpctl/pull/526)
- Implemented DPCTLQueue_SubmitBarrier, DPCTLQueue_SubmitBarrierForEvents,
SyclQueue.submit_barrier [#524](https://github.com/IntelPython/dpctl/pull/524)
- Added C-API DPCTLQueue_HasEnableProfiling
- Added Python API SyclQueue.has_enable_profiling
- Use public for data owning class definitions
- Queue has enable profiling [#531](https://github.com/IntelPython/dpctl/pull/531)
- Use public for data owning class definitions [#533](https://github.com/IntelPython/dpctl/pull/533)
- Added logic to verify that all bits of property integer were recognized and used [#494](https://github.com/IntelPython/dpctl/pull/494)
- Added support for some properties/methods of underluing device
- A test for properties, method of q mirroring that of device
- Conda build scripts should build wheels in the same setup invocation as install [#538](https://github.com/IntelPython/dpctl/pull/538)
- Added install_requires keyword to setup call
- Added requirements.txt files in dpctl/ and in dpctl/docs [#540](https://github.com/IntelPython/dpctl/pull/540)
- Improved C-API for dpctl Cython classes, added example of using them in Pybind11 extension. [#550](https://github.com/IntelPython/dpctl/pull/550)
- dpctl.SyclEvent acquired ability to get command status and get profiling information. [#553](https://github.com/IntelPython/dpctl/pull/553)

### Changed
- Moved DPCLSyclInterface library from MANIFEST.in [#482](https://github.com/IntelPython/dpctl/pull/482)
- Refactored tests
- Use dpcpp compiler package for Linux [#514](https://github.com/IntelPython/dpctl/pull/514)
- Update conda-package.yml
- Static methods _init_helper made into functions and removed from PXD files [#532](https://github.com/IntelPython/dpctl/pull/532)

### Removed
- Remove imports from __future__ [#485](https://github.com/IntelPython/dpctl/pull/485)

### Fixed
- Fix sub devices [#479](https://github.com/IntelPython/dpctl/pull/479)
- Fix addressof_ref function in `SyclContext` [#488](https://github.com/IntelPython/dpctl/pull/488)
- Follow `DPCTLDevice_CreateFromSelector` which passes the check [#487](https://github.com/IntelPython/dpctl/pull/487)
- Fix a typo in the pytest configuration [#490](https://github.com/IntelPython/dpctl/pull/490)
- Fixed dbg_build.sh script for Linux to use L0
- Reuse IntelSycl_LIBRARY_DIR variable in cmake
- CXX, dpcpp used on Windows too
- Update conda-recipe/bld.bat
- Change to SyclQueue.__repr__ to reflect properties [#531](https://github.com/IntelPython/dpctl/pull/531)
- Static methods `_init_helper` made into functions and removed from PXD files [#532](https://github.com/IntelPython/dpctl/pull/532)
- Fixed typo in pip installation instruction [#536](https://github.com/IntelPython/dpctl/pull/536)
- Fixed dpctl_config.h, added dpctl_service.h, .cpp [#539](https://github.com/IntelPython/dpctl/pull/539)
- Fixed `__sycl_usm_array_interface__` output for 0d arrays [#547](https://github.com/IntelPython/dpctl/pull/547)

## [0.8.0] - 05/26/2021

### Added
- Implemented support for constructing MemoryUSM* from object with
__sycl_usm_array_interface__ when array-info is not contiguous [#400](https://github.com/IntelPython/dpctl/pull/400)
- Print the backend as part of SyclDevice.print_device_info function [#409](https://github.com/IntelPython/dpctl/pull/409)
- Added dpctl/tensor/_usmarray submodule [#427](https://github.com/IntelPython/dpctl/pull/427)
- Added arg checking to functions in dpctl_sycl_usm_interface.cpp [#430](https://github.com/IntelPython/dpctl/pull/430)
- A static method of _Memory to create from external allocation [#430](https://github.com/IntelPython/dpctl/pull/430)
- Added usm_ndarray accessors [#435](https://github.com/IntelPython/dpctl/pull/435)
- Added Device class representing Data-API notion of device [#440](https://github.com/IntelPython/dpctl/pull/440)
- Added free Python function as_usm_memory(obj) [#443](https://github.com/IntelPython/dpctl/pull/443) and associated unit
tests [#449](https://github.com/IntelPython/dpctl/pull/449)
- Dependency for numpy 1.17 [#445](https://github.com/IntelPython/dpctl/pull/445)
- Add a flag to make doxygen HTML generation optional [#450](https://github.com/IntelPython/dpctl/pull/450)
- Added a feature to get the filter string for a device from Python using the
new dpctl.SyclDevice.get_filter_string method. Also added the corresponding
DPCTLDeviceMgr_GetPositionInDevices(DRef, device_mask) C API function [#453](https://github.com/IntelPython/dpctl/pull/453)
- New options to setup.py to specify which dpcpp compiler to use, if L0
program creation is to be supported, and to generate code coverage [#426](https://github.com/IntelPython/dpctl/pull/426)
- Github action to check Python code quality [#422](https://github.com/IntelPython/dpctl/pull/422)
- Github action to auto-publish Sphinx docs for master [#446](https://github.com/IntelPython/dpctl/pull/446)
- Github action to generate coverage report and publish to coveralls.io [#459](https://github.com/IntelPython/dpctl/pull/459)

### Changed
- Rename dpctl.dptensor to dpctl.tensor [#407](https://github.com/IntelPython/dpctl/pull/407)
- Changed repr for Memory objects [#442](https://github.com/IntelPython/dpctl/pull/442)
- Used dpctl.SyclQueue instead of manager and get current queue in tests for
SyclProgram [#448](https://github.com/IntelPython/dpctl/pull/448)
-

### Fixed
- Issue #189 dpctl.memory.MemoryUSMShared(np.int64(16)) should work [#392](https://github.com/IntelPython/dpctl/pull/392)
- Use size_t instead of Py_ssize_t to fit device USM pointer [#405](https://github.com/IntelPython/dpctl/pull/405)
- Various code quality issues identified by flake8 (#417, #419, #420, #422)
- Fixed issues in slicing and array construction [#441](https://github.com/IntelPython/dpctl/pull/441)
- Fixed an issue [#447](https://github.com/IntelPython/dpctl/pull/447) where dpctl.get_devices does not return devices in the
same order as sycl::device::get_devices [#451](https://github.com/IntelPython/dpctl/pull/451)
- L0 program creation support on Windows [#319](https://github.com/IntelPython/dpctl/pull/319)

### Removed
- Removing public keyword to get_current_queue Cython declaration [#437](https://github.com/IntelPython/dpctl/pull/437)

## [0.7.0] - 05/03/2021
### Added
- Complete support for `sycl::ONEAPI::filter_selector` in dpctl.
, and `sycl::platform` [#298](https://github.com/IntelPython/dpctl/pull/298)
creation using opaque pointers.
- A `DPCTLDeviceMgr` module in C API that caches a default context for root
devices [#277](https://github.com/IntelPython/dpctl/pull/277).
- `DPCTLSyclBackendType` and `DPCTLSyclDeviceType` have a new member `ALL`
[#287](https://github.com/IntelPython/dpctl/pull/287).
- C API now provides helper functions to convert between dpctl and SYCL enum
values [#296](https://github.com/IntelPython/dpctl/pull/296).
- Macros to help create opaque vector classes for opaque SYCL types [#297](https://github.com/IntelPython/dpctl/pull/297).
, `SyclContext` [#334](https://github.com/IntelPython/dpctl/pull/334), `SyclPlatform` (#336, #298),
`SyclQueue` [#323](https://github.com/IntelPython/dpctl/pull/323) have constructors that recognize filter selectors and closely
follow DPC++ interface.
- Add API to get a `PyCapsule` from `SyclQueue`, `SyclContext` instances [#350](https://github.com/IntelPython/dpctl/pull/350).
- Added `get_queue_ref_from_ptr_and_syclobj(ptr, syclobj)` that creates
`DPCTLSyclQueueRef` from a USM pointer and Python object `syclobj` from
`__sycl_usm_array_interface__` [#380](https://github.com/IntelPython/dpctl/pull/380).
- Support for SYCL sub-devices, including sub-device creation, queue, and
context creation using sub-devices [#343](https://github.com/IntelPython/dpctl/pull/343).
- `SyclDevice.parent_device` property to indicate if an instance has a parent
device [#366](https://github.com/IntelPython/dpctl/pull/366).
- Several new getter functions for device info descriptors to device interface
(#300, #335, #318, #315, #308).
- Support for SYCL device aspects [#307](https://github.com/IntelPython/dpctl/pull/307).
- Properties for every `sycl::device` info and aspect that we support in
`SyclDevice` [#324](https://github.com/IntelPython/dpctl/pull/324).
- Support handling async errors inside `SylQueue` instances [#346](https://github.com/IntelPython/dpctl/pull/346).
- `get_backend`, `get_platform`, `get_device_type` to Python `SyclDevice` class [#300](https://github.com/IntelPython/dpctl/pull/300)
- A `_sycl_device_factory.pyx` module providing `SyclDevice` constructors using
standard `sycl::device_selector` classes (previously in `_sycl_device.pyx`)
and a new `get_devices` [#277](https://github.com/IntelPython/dpctl/pull/277) function to enumerate all devices.
- `_sycl_device_factory.pyx` implements `get_num_devices` and `has_*_device(s)`
functions [#320](https://github.com/IntelPython/dpctl/pull/320).
- Enable Python coverage in CI for Linux [#369](https://github.com/IntelPython/dpctl/pull/369).
- Use `public` keyword in `_sycl_*.pxd` to generate header files allowing
non-Cython centric native extensions to work with dpctl's Python objects
[#218](https://github.com/IntelPython/dpctl/pull/218).
- Documentation improvements [#341](https://github.com/IntelPython/dpctl/pull/341).

### Changed
- Rename dpCtl to dpctl in all comments, license headers, and docs. [#342](https://github.com/IntelPython/dpctl/pull/342)
- `dpctl.memory.MemoryUSM*` constructors now use `dpctl.SyclQueue()` instead of
`dpctl.get_current_queue()` when the `queue` keyword argument is `None` (default) [#382](https://github.com/IntelPython/dpctl/pull/382).
- `dpctl.set_default_queue` has been renamed to `dpctl.set_global_queue()` [#323](https://github.com/IntelPython/dpctl/pull/323).
- Changed `dpctl.dump` to `dpctl.lsplatform` [#336](https://github.com/IntelPython/dpctl/pull/336).
- Various `SyclDevice` methods related to querying `sycl::info::device` were converted
to properties [#324](https://github.com/IntelPython/dpctl/pull/324).
- Various C API functions names were changed.

### Fixed
- Possible crashes when a SYCL platform is not available [#349](https://github.com/IntelPython/dpctl/pull/349).
- Fix tests which fail if GPU is not available (only CPU is available) [#359](https://github.com/IntelPython/dpctl/pull/359).
- Fix breaking C API tests [#358](https://github.com/IntelPython/dpctl/pull/358).
- Bandit warning about "subprocess.check_call(shell=True)" for Windows [#306](https://github.com/IntelPython/dpctl/pull/306).

### Removed
- Removed `get_num_platforms`, `has_cpu_queues`, `has_gpu_queues`, `get_num_queues`,
`has_sycl_platforms` [#320](https://github.com/IntelPython/dpctl/pull/320).

## [0.6.1] - 2021-03-01
### Fixed
- Do not use POP_FRONT in FindDPCPP.cmake so that we can use a cmake version older that 3.15.

## [0.6.0] - 2021-03-01
### Added
- Documentation improvements.
- Cmake improvements and Coverage for C API, Cython and Python.
- Added support for Level Zero devices and queues.
- Added support for SYCL standard device_selector classes.
- SyclDevice instances can now be constructed using filter selector strings.
- Code of conduct.
- Building wheels.
- Queue manager improvements.
- Adding `__array_function__` so that Numpy calls with dparrays work.
- Using clang-format for C/C++ code formatting.
- Using pytest for running tests.
- Add python and cython file coverage.
- Using Bandit for finding common security issues in Python code.
- Add instructions about file headers formats.

### Changed
- Changed compiler name usage from clang++ to dpcpp.
- Reformat backend.pxd to be closer to black style.

### Fixed
- Remove `cython` from `install_requires`. It allows use `dpCtl` in `numba` extensions.
- Incorrect import in example.
- Consistency of file headers.
- Klocwork issues.

## [0.5.0] - 2020-12-17
### Added
- `_Memory.get_pointer_type` static method which returns kind of USM pointer.
- Utility functions to transform string to device type and back.
- New `dpctl.dptensor.numpy_usm_shared` module containing USM array. USM array
extends NumPy ndarray.
- A lot of new examples. Including examples of building Cython extensions with DPC++ compiler that interoperate with dpCtl.
- Mechanism for registering a callback function to look and see if the object
supports USM.

### Changed
- setup.py builds C++ backend for develop and install commands.
- Building wheels.
- Use DPC++ runtime from package `dpcpp_cpp_rt`.
- All usage of `DPPL` in C-API functions was changed to `DPCTL`, _e.g._, `DPPLQueueMgr_GetCurrentQueue` to `DPCTLQueueMgr_GetCurrentQueue`.
- Renamed the C-API directory is now called `dpctl-capi` instead of `backends`.
- Refactoring the `dpctl-capi` functions to prepare for changes to add Level Zero program creation.
- `SyclProgram` and `SyclKernel` classes were moved out of `dpctl` into the `dpctl.program` sub-module.

### Fixed
- Klockwork static code analysis warnings.

## [0.4.0] - 2020-11-04
### Added
- Device descriptors "max_compute_units", "max_work_item_dimensions", "max_work_item_sizes", "max_work_group_size", "max_num_sub_groups" and "aspects" for int64 atomics inside dpctl C API and inside the dpctl.SyclDevice class.
- MemoryUSM* classes moved to `dpctl.memory` module, added support for aligned allocation, added support for `prefetch` and `mem_advise` (sychronous) methods, implemented `copy_to_host`, `copy_from_host` and `copy_from_device` methods, pickling support, and zero-copy interoperability with Python objects which implement `__sycl_usm_array_inerface__` protocol.
- Helper scripts to generate API documentation for both C API and Python.

### Fixed
- Compiler warnings when building libDPPLSyclInterface and the Cython extensions.

### Removed
- The Legacy OpenCL interface.

## [0.3.8] - 2020-10-08

### Changed
- How the initial active queue is populated inside DPPLQueueMgr.
- dpctl.SyclQueueManager only reports the number of non-host platform.
- dpctl.SyclQueueManager now raises an exception if DPCTL C API returns a nullptr instead of a valid Sycl queue.

### Fixed
- Several crashes in cases where an OpenCL or Level Zero platform is not available.
- Fix failing platform test case. [#116](https://github.com/IntelPython/dpctl/pull/116)
- Properly skip tests when no OpenCL devices are available.
- Add skip tests to test_sycl_usm.py
- Fix Gtests configuration.

## [0.3.7] - 2020-10-08
### Fixed
- A crash on Windows due a Level Zero driver problem. Each device was getting enumerated twice. To handle the issue, we added a temporary fix to use only first device for each device type and backend [#118](https://github.com/IntelPython/dpctl/pull/118).

## [0.3.6] - 2020-10-06
### Added
- Changelog was added for dpctl.

### Fixed
- Windows build was fixed.

## [0.3.5] - 2020-10-06
### Added
- Add a helper function to all Python SyclXXX classes to get the address of the base C API pointer as a long.

### Changed
- Rename PyDPPL to dpCtl in comments (function name renaming to come later)

### Fixed
- Fix bugs highlighted by tools.
- Various code clean ups.

## [0.3.4] - 2020-10-05
### Added
- Dump functions were enhanced to print back-end information.
- dpctl gained support for unint_8 and unsigned long data types.
- oneAPI Beta 10 tool chain support was added.

### Changed
- dpctl is now aware of DPC++ Sycl PI back-ends. The functionality is now exposed via the context interface.
- C API's queue manager was refactored to require back-end.
- dpct's device_context now requires back-end, device-type, and device-id to be provided in a string format, e.g. opencl:gpu:0.

### Fixed
- Fixed some important bugs found by static analysis.

## [0.3.3] - 2020-10-02
### Added
- Add dpctl.get_curent_device_type().

## [0.3.2] - 2020-09-29
### Changed
- Set _cpu_device and _gpu_device to None by default.

## [0.3.1] - 2020-09-28
### Added
- Add get include and include headers.

### Changed
- DPPL shared objects are installed into dpctl.

### Fixed
- Refactor unit tests.

## [0.3.0] - 2020-09-23
### Added
- Adds C and Cython API for portions of Sycl queue, device, context interfaces.
- Implementing USM memory management.

### Changed
- Refactored API to expose a minimal sycl::queue interface.
- Modify cpu_queues, gpu_queues and active_queues to functions.
- Change static vectors to static pointers to verctors. It disables call for destructors. Destructors are also call in undefined order.
- Rename package PyDPPL to dpCtl.
- Use dpcpp.exe on Windows instead of dpcpp-cl.exe deleted in oneAPI beta08.

### Fixed
- Correct use ERRORLEVEL in conda scripts for Windows.
- Fix using dppl.has_sycl_platforms() and dppl.has_gpu_queues() functions in skipIf
