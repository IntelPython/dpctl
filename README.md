[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Coverage Status](https://coveralls.io/repos/github/IntelPython/dpctl/badge.svg?branch=master)](https://coveralls.io/github/IntelPython/dpctl?branch=master)
![Generate Documentation](https://github.com/IntelPython/dpctl/actions/workflows/generate-docs.yml/badge.svg?branch=master)
[![Join the chat at https://matrix.to/#/#Data-Parallel-Python_community:gitter.im](https://badges.gitter.im/Join%20Chat.svg)](https://app.gitter.im/#/room/#Data-Parallel-Python_community:gitter.im)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/IntelPython/dpctl/badge)](https://securityscorecards.dev/viewer/?uri=github.com/IntelPython/dpctl)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8435/badge)](https://www.bestpractices.dev/projects/8435)

<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo" width="75"/>

# Data Parallel Control

Data Parallel Control or `dpctl` is a Python library that allows users
to *control* the execution placement of a [compute
kernel](https://en.wikipedia.org/wiki/Compute_kernel) on an
[XPU](https://www.intel.com/content/www/us/en/newsroom/news/xpu-vision-oneapi-server-gpu.html).

The compute kernel can be a code:
* written by the user, e.g., using [`numba-dpex`](https://github.com/IntelPython/numba-dpex)
* that is part of a library, such as oneMKL

The `dpctl` library is built upon the [SYCL
standard](https://www.khronos.org/sycl/). It implements Python
bindings for a subset of the standard [runtime
classes](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_runtime_classes) that allow users to:
* query platforms
* discover and represent devices and sub-devices
* construct contexts and queues

`dpctl` features classes for [SYCL Unified Shared Memory
(USM)](https://link.springer.com/chapter/10.1007/978-1-4842-5574-2_6)
management and implements a tensor library conforming to [Python Array
API](https://data-apis.org/array-api/latest/) standard.

The library helps authors of Python native extensions written
in C, Cython, or pybind11 to access `dpctl` objects representing SYCL
devices, queues, memory, and tensors.

`Dpctl` is the core part of a larger family of [data-parallel Python
libraries and tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
to program on XPUs.



# Installing

You can install the library using [conda](https://anaconda.org/intel/dpctl) or
[pip](https://pypi.org/project/dpctl/) package managers. It is also available in the [Intel(R)
Distribution for
Python](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
(IDP).

## Intel(R) oneAPI

You can find the most recent release of `dpctl` every quarter as part of the Intel(R) oneAPI releases.

To get the library from the latest oneAPI release, follow the
instructions from Intel(R) [oneAPI installation
guide](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html).

> **NOTE:** You need to install the Intel(R) oneAPI AI Analytics Tookit to get
>IDP and `dpctl`.


## Conda

To install `dpctl` from the Intel(R) channel on Anaconda
cloud, use the following command:

```bash
conda install dpctl -c intel
```

## Pip

The `dpctl` can be installed using `pip` obtaining wheel packages either from PyPi or from Intel(R) channel on Anaconda.
To install `dpctl` wheel package from Intel(R) channel on Anaconda, run the following command:

```bash
python -m pip install --index-url https://pypi.anaconda.org/intel/simple dpctl
```

Installing the bleeding edge
------------------------

To try out the latest features, install `dpctl` from our
development channel on Anaconda cloud:

```bash
conda install dpctl -c dppy/label/dev
```

# Building

Refer to our [Documentation](https://intelpython.github.io/dpctl) for more information on
setting up a development environment and building `dpctl` from the source.


# Examples

Our examples are located in the [examples/](examples) folder and are organized in sub-folders. Examples
in the [Python/](examples/python) folder demonstrate how to inspect the heterogeneous platform,
select a device, create an execution queue, and how to control device memory allocation and
execution placement.

Examples in [Cython/](examples/cython), [C/](examples/c), and [Pybind11](examples/pybind11) folders
demonstrate creation of SYCL-powered native Python extensions. Please refer to each folder's README
document for directions on how to build and use each example.


# Running Tests

Tests are located in folder [dpctl/tests](dpctl/tests).

To run the tests, use:
```bash
pytest --pyargs dpctl
```

Running full test suite requires working C/C++ compiler. To run the test suite without one, use:

```bash
pytest --pyargs dpctl -k "not test_cython_api and not test_c_headers"
```
