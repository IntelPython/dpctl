# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.2] - 11/xx/2021

## Fixed
- Fixed issue #649 about incorrect behavior of `.T` method on sliced arrays (#653)

## [0.11.1] - 11/10/2021

### Changed
- Replaced uses of clang compiler with icx executable (#665)

## [0.11.0] - 11/01/2021

### Added
- Use Python 3.9 in public CI (#599)
- Add a new C API utility function (`DPCTLDeviceMgr_GetDeviceInfoStr`) to return the device info as a C string object (#620)
- New Github workflow to build dpclt with nightly Intel llvm/sycl + drivers (#621)
- Always raise SubDeviceCreationError even when sub-device counts are zero (#622)
- Updated OpenCL interoprability code to fix build with Intel llvm/sycl bundle (#625)
- Enabled use of default platform context extension in SYCL compilers that implement this extension (#627)
- Implemented `dpctl.utils.get_execution_queue(queue_seq)` utility to help implementing "compute-follows data" convention for offload target (#632)
- Improved code coverage (#619) (#542) (#631)

### Changed
- Replaced `host_device` device type with `host` in tests (#616)
- Rework the logic in `dpctl.memory`'s `copy_from_device` method to work correctly with `host` device (#618)
- Use `dpctl.device_type.host` instead of `dpctl.device_type.host_device` (#626)
- Reinstate deprecated `sycl::program` and that was conditionally removed from open source DPC++ toolchain (#633)
- Use `LoadLibraryExA` instead of `LoadLibraryA` to mitigate a possible DLL injection issue when we load the Level zero DLL on windows (#636)
- Github coverage workflow is changed to use oneAPI 2021.3 instead of latest to work around broken profiling instrumentation in DPC++ 2021.4 (#614)
- Update build dependencies for NumPy (#641)
- Use "readelf" on SYCL's `pi_level_zero` library to find out and use the exact name of `ze_loader.so` in SyclInterface library (#617)

### Removed
- Removed use of DPC++ features deprecated in 2021.4 and open source Intel llvm/sycl compiler (#603)

### Fixed
- Suppress errant CMake log (#610)
- Fixes to compile dpctl using Intel llvm/sycl compiler (#603)
- Fix for the hang is to avoid passing `nullptr` argument to `sycl::queue::prefetch` (#612)
- Fixed the logic to return device count (#623)
- Enabled building of C extensions with dpctl by including header defining `bool` type for C compilers (#604)

## [0.10.0] - 09/28/2021

### Added
- Added methods __bool__, __float__, __int__, __index__,
  and __complex__ to usm_ndarray (#578)
- Added data-API required special methods to usm_ndarray class,
  as well as to_numpy/from_numpy, astype, reshape functions (#586)
- Added methods to query dpctl.SyclDevice for size of global/local memory (#589)
- Added tests for constructors with invalid capsules (#577)
- Improved test coverage of `dpctl.SyclQueue` implementation (#574)
- Added a test to exercise API exported function (get_event_ref). (#570)
- Expanded tests in test_sycl_context to improve coverage (#571)
- Tweaks to test_sycl_event to improve coverage (#567)
- Improved coverage of dpctl.__init__ file and other service functions (#563)
- Added test for repr and test for default argument to constructor (#565)
- Added some tests to involve capsule (#564)
- Added workflow for Public CI on Windows (#534)
- DPCTLQueue_Memcpy, _Prefetch, _Memadvise become asynchronous (#557)
- Added device aspect selector, `dpctl.select_device_with_aspects` (#558)
- Added test based on example from #583

### Changed
- Parametrized tests for executing OpenCL kernels compiled from source in types of arguments (#581)
- Temporary disabled self-hosted CI jobs runner (#559)
- Changed static method `SyclQueue._create_from_context_and_device` (#579)
- Transitioned all Python API to use pytest over unittest, improved coverage in dpctl/memory (#575)
- Changed `dpctl.SyclEvent.profiling_info_submit` from method to a property (#573)
- Simplified arg parsing in SyclDevice constructor (#572)
- Used<img> tag with alignment attribute set in README (#562)
- Moved sycl timer into dpctl.SyclTimer (#555)
- Used clang-format off, clang-format on to avoid include reordering in pybind11 example (#588)

### Fixed
- Implemented a workaround for running conda-build using Klocwork (#566)
- Separated pipelines for Linux and Windows (#582)
- Fixed inconsistency in `__sycl_usm_array_interface__` of `usm_ndarray` instance (#584)
- Fixed memory leak: Capsule deleters now free resources for renamed capsules too (#568)
- Fixed __version__ test to allow for semantic versioning (#569)
- Improved coverage of _types.pxi (#556)
- Fixed `UnboundLocalError` when default queue could not be created (#554)

## [0.9.0] - 08/25/2021

### Added
- Improvements to logic for working with custom DPC++ toolchain (#481)
- Add SyclContext unit test cases (#488)
- Consolidate configurations of tools that support PEP 518 into pyproject.toml (#486)
- Added C-API hash function, used them in Python interface (#491)
- Add missing extra checks to ensure unwrapped pointer is not Null
- Add error messages to L0 program creation routine
- Improve test coverage for dpctl_sycl_queue_interface (#492)
- Use pytest.warns in test_lsplatform3 (#495)
- Added test class to test DRef=nullptr case (#496)
- Extend parameterized test in test_sycl_queue_interface (#497)
- Use Memcpy, memadvise in tests
- Expanded types tests by TestQueueSubmitRange
- Added a test that retrieved DPCPP compiled kernel and submits them via DPCTLQueue_SubmitRange (#499)
- Add DPCTLEvent_Copy (#504), DPCTLEvent_GetBackend (#507), DPCTLEvent_GetCommandExecutionStatus (#516),
  DPCTLEvent_WaitAndThrow (#513), DPCTLEvent_GetWaitList (#510) functions
- Propagate compile flags (#512)
- Add conda package CI pipeline on GitHub Actions (#515)
- Run tests on GPU (#518)
- Add 3 wrapper func for event::get_profiling_info (#519)
- Changes to build_backend.py to enable sycl-compiler-prefix on Windows
- dtype keyword of usm_ndarray now supports np.double and other types (#526)
- Implemented DPCTLQueue_SubmitBarrier, DPCTLQueue_SubmitBarrierForEvents,
  SyclQueue.submit_barrier (#524)
- Added C-API DPCTLQueue_HasEnableProfiling
- Added Python API SyclQueue.has_enable_profiling
- Use public for data owning class definitions
- Queue has enable profiling (#531)
- Use public for data owning class definitions (#533)
- Added logic to verify that all bits of property integer were recognized and used (#494)
- Added support for some properties/methods of underluing device
- A test for properties, method of q mirroring that of device
- Conda build scripts should build wheels in the same setup invocation as install (#538)
- Added install_requires keyword to setup call
- Added requirements.txt files in dpctl/ and in dpctl/docs (#540)
- Improved C-API for dpctl Cython classes, added example of using them in Pybind11 extension. (#550)
- dpctl.SyclEvent acquired ability to get command status and get profiling information. (#553)

### Changed
- Moved DPCLSyclInterface library from MANIFEST.in (#482)
- Refactored tests
- Use dpcpp compiler package for Linux (#514)
- Update conda-package.yml
- Static methods _init_helper made into functions and removed from PXD files (#532)

### Removed
- Remove imports from __future__ (#485)

### Fixed
- Fix sub devices (#479)
- Fix addressof_ref function in `SyclContext` (#488)
- Follow `DPCTLDevice_CreateFromSelector` which passes the check (#487)
- Fix a typo in the pytest configuration (#490)
- Fixed dbg_build.sh script for Linux to use L0
- Reuse IntelSycl_LIBRARY_DIR variable in cmake
- CXX, dpcpp used on Windows too
- Update conda-recipe/bld.bat
- Change to SyclQueue.__repr__ to reflect properties (#531)
- Static methods `_init_helper` made into functions and removed from PXD files (#532)
- Fixed typo in pip installation instruction (#536)
- Fixed dpctl_config.h, added dpctl_service.h, .cpp (#539)
- Fixed `__sycl_usm_array_interface__` output for 0d arrays (#547)

## [0.8.0] - 05/26/2021

### Added
- Implemented support for constructing MemoryUSM* from object with
  __sycl_usm_array_interface__ when array-info is not contiguous (#400)
- Print the backend as part of SyclDevice.print_device_info function (#409)
- Added dpctl/tensor/_usmarray submodule (#427)
- Added arg checking to functions in dpctl_sycl_usm_interface.cpp (#430)
- A static method of _Memory to create from external allocation (#430)
- Added usm_ndarray accessors (#435)
- Added Device class representing Data-API notion of device (#440)
- Added free Python function as_usm_memory(obj) (#443) and associated unit
  tests (#449)
- Dependency for numpy 1.17 (#445)
- Add a flag to make doxygen HTML generation optional (#450)
- Added a feature to get the filter string for a device from Python using the
  new dpctl.SyclDevice.get_filter_string method. Also added the corresponding
  DPCTLDeviceMgr_GetPositionInDevices(DRef, device_mask) C API function (#453)
- New options to setup.py to specify which dpcpp compiler to use, if L0
  program creation is to be supported, and to generate code coverage (#426)
- Github action to check Python code quality (#422)
- Github action to auto-publish Sphinx docs for master (#446)
- Github action to generate coverage report and publish to coveralls.io (#459)

### Changed
- Rename dpctl.dptensor to dpctl.tensor (#407)
- Changed repr for Memory objects (#442)
- Used dpctl.SyclQueue instead of manager and get current queue in tests for
  SyclProgram (#448)
-

### Fixed
- Issue #189 dpctl.memory.MemoryUSMShared(np.int64(16)) should work (#392)
- Use size_t instead of Py_ssize_t to fit device USM pointer (#405)
- Various code quality issues identified by flake8 (#417, #419, #420, #422)
- Fixed issues in slicing and array construction (#441)
- Fixed an issue (#447) where dpctl.get_devices does not return devices in the
  same order as sycl::device::get_devices (#451)
- L0 program creation support on Windows (#319)

### Removed
- Removing public keyword to get_current_queue Cython declaration (#437)

## [0.7.0] - 05/03/2021
### Added
- Complete support for `sycl::ONEAPI::filter_selector` in dpctl.
- C API for `sycl::queue` (#323), `syc::context` (#331), and `sycl::platform` (#298)
  creation using opaque pointers.
- A `DPCTLDeviceMgr` module in C API that caches a default context for root
  devices (#277).
- `DPCTLSyclBackendType` and `DPCTLSyclDeviceType` have a new member `ALL`
  (#287).
- C API now provides helper functions to convert between dpctl and SYCL enum
  values (#296).
- Macros to help create opaque vector classes for opaque SYCL types (#297).
- `SyclDevice` (#321), `SyclContext` (#334), `SyclPlatform` (#336, #298),
  `SyclQueue` (#323) have constructors that recognize filter selectors and closely
  follow DPC++ interface.
- Add API to get a `PyCapsule` from `SyclQueue`, `SyclContext` instances (#350).
- Added `get_queue_ref_from_ptr_and_syclobj(ptr, syclobj)` that creates
  `DPCTLSyclQueueRef` from a USM pointer and Python object `syclobj` from
  `__sycl_usm_array_interface__` (#380).
- Support for SYCL sub-devices, including sub-device creation, queue, and
  context creation using sub-devices (#343).
- `SyclDevice.parent_device` property to indicate if an instance has a parent
  device (#366).
- Several new getter functions for device info descriptors to device interface
  (#300, #335, #318, #315, #308).
- Support for SYCL device aspects (#307).
- Properties for every `sycl::device` info and aspect that we support in
  `SyclDevice` (#324).
- Support handling async errors inside `SylQueue` instances (#346).
- `get_backend`, `get_platform`, `get_device_type` to Python `SyclDevice` class (#300)
- A `_sycl_device_factory.pyx` module providing `SyclDevice` constructors using
  standard `sycl::device_selector` classes (previously in `_sycl_device.pyx`)
  and a new `get_devices` (#277) function to enumerate all devices.
- `_sycl_device_factory.pyx` implements `get_num_devices` and `has_*_device(s)`
  functions (#320).
- Enable Python coverage in CI for Linux (#369).
- Use `public` keyword in `_sycl_*.pxd` to generate header files allowing
  non-Cython centric native extensions to work with dpctl's Python objects
  (#218).
- Documentation improvements (#341).

### Changed
- Rename dpCtl to dpctl in all comments, license headers, and docs. (#342)
- `dpctl.memory.MemoryUSM*` constructors now use `dpctl.SyclQueue()` instead of
  `dpctl.get_current_queue()` when the `queue` keyword argument is `None` (default) (#382).
- `dpctl.set_default_queue` has been renamed to `dpctl.set_global_queue()` (#323).
- Changed `dpctl.dump` to `dpctl.lsplatform` (#336).
- Various `SyclDevice` methods related to querying `sycl::info::device` were converted
  to properties (#324).
- Various C API functions names were changed.

### Fixed
- Possible crashes when a SYCL platform is not available (#349).
- Fix tests which fail if GPU is not available (only CPU is available) (#359).
- Fix breaking C API tests (#358).
- Bandit warning about "subprocess.check_call(shell=True)" for Windows (#306).

### Removed
- Removed `get_num_platforms`, `has_cpu_queues`, `has_gpu_queues`, `get_num_queues`,
  `has_sycl_platforms` (#320).

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
  - Fix failing platform test case. (#116)
  - Properly skip tests when no OpenCL devices are available.
  - Add skip tests to test_sycl_usm.py
  - Fix Gtests configuration.

## [0.3.7] - 2020-10-08
### Fixed
- A crash on Windows due a Level Zero driver problem. Each device was getting enumerated twice. To handle the issue, we added a temporary fix to use only first device for each device type and backend (#118).

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
