# Examples of using `dpctl`

The `dpctl` is a foundational package facilitating use of [SYCL](sycl) to extend Python's reach to heterogeneous systems.

## Python

The `dpctl` provides Python API to SYCL runtime permitting user to
inspect the heterogeneous [platform],
[select](device_seelection) amongst available devices,
query [properties](device_descriptors) of created devices,
and construct [queues] to specify execution placement of offloaded computation.

Examples of this functionality are located in the [python](python) folder.

## Cython

The `dpctl` integrates with [Cython], a native extension generator, to facilitate building
SYCL-powered Python extensions.

Examples of Python extensions written using Cython are located in the [cython](cython) folder.

## Pybind11

Since [SYCL](sycl) is based on C++, [pybind11] is a natural tool of choice to author SYCL-powered
Python extensions. The `dpctl` provides `dpctl4pybind11.hpp` integration header to provide natural
mapping between SYCL C++ classes and `dpctl` Python types.

Examples of Python extensions created with `pybind11` are located in the [pybind11](pybind11) folder.

## C

The `dpctl` implements `DPCTLSyclInterface` C library and C-API to work with Python objects and types
implemented in `dpctl`. Use integration headers `dpctl_sycl_interface.h` and `dpctl_capi.h` to access
this functionality.

Examples of Python extensions created using C are located in [c](c) folder.

## Building with AdaptiveCpp

An extension must be built with the same SYCL implementation as the `dpctl` it is
built against, which `dpctl._diagnostics.sycl_provider()` reports. The `cython`
and `pybind11` examples accept `-DDPCTL_SYCL_PROVIDER=AdaptiveCpp` to build with
AdaptiveCpp rather than Intel DPC++:

```bash
CC=clang CXX=acpp python setup.py build_ext --inplace -G Ninja -- \
    -DDPCTL_SYCL_PROVIDER=AdaptiveCpp
```

The `onemkl_gemv` and `use_dpctl_sycl_kernel` examples require Intel DPC++, see
their READMEs.


[platform]: https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/platforms.html
[device_selection]: https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/device_selection.html
[device_descriptors]: https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/devices.html#device-aspects-and-information-descriptors
[queues]: https://intelpython.github.io/dpctl/latest/docfiles/user_guides/manual/dpctl/queues.html
[array_api]: https://data-apis.org/array-api/
[sycl]: https://registry.khronos.org/SYCL/
[Cython]: https://cython.org/
[pybind11]: https://pybind11.readthedocs.io
