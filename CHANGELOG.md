# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Device descriptors "max_compute_units", "max_work_item_dimensions", "max_work_item_sizes", "max_work_group_size", "max_num_sub_groups" and "aspects" for int64 atomics inside dpctl C API and inside the dpctl.SyclDevice class.
- MemoryUSM* classes moved to `dpctl.memory` module, added support for aligned allocation, added support for `prefetch` and `mem_advise` (sychronous) methods, implemented `copy_to_host`, `copy_from_host` and `copy_from_device` methods, pickling support, and zero-copy interoperability with Python objects which implement `__sycl_usm_array_inerface__` protocol.

### Fixed
- Compiler warnings when building libDPPLSyclInterface and the Cython extensions.

### Removed
- The Legacy OpenCL interface.

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
