[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Coverage Status](https://coveralls.io/repos/github/IntelPython/dpctl/badge.svg?branch=master)](https://coveralls.io/github/IntelPython/dpctl?branch=master)
![Generate Documentation](https://github.com/IntelPython/dpctl/actions/workflows/generate-docs.yml/badge.svg?branch=master)


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
standard](https://www.khronos.org/sycl/). It also implements Python
bindings for a subset of the standard [runtime
classes](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_runtime_classes) that allow users to:
* query platforms
* discover and represent devices and sub-devices
* construct contexts and queues

`dpctl` features classes for [SYCL Unified Shared Memory
(USM)](https://link.springer.com/chapter/10.1007/978-1-4842-5574-2_6)
management and implements a tensor [array
API](https://data-apis.org/array-api/latest/).

The library helps authors of Python native extensions written
in C, Cython, or pybind11 to access `dpctl` objects representing SYCL
devices, queues, memory, and tensors.

`Dpctl` is the core part of a larger family of [data-parallel Python
libraries and tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)
to program on XPUs.



# Installing

You can install the library with [conda](https://anaconda.org/intel/dpctl) and
[pip](https://pypi.org/project/dpctl/). It is also available in the [Intel(R)
Distribution for
Python](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-python.html)
(IDP).

## Inte(R) oneAPI

You can find the most recent release of `dpctl` every quarter as part of the Intel(R) oneAPI releases.

To get the library from the latest oneAPI release, follow the
instructions from Intel(R) [oneAPI installation
guide](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html).

> **NOTE:** You need to install the Intel(R) oneAPI Basekit to get
>IDP and `dpctl`.


## Conda

To install `dpctl` from the Intel(R) channel on Anaconda
cloud, use the following command:

```bash
conda install dpctl -c intel
```

## PyPi

To install `dpctl` from PyPi, run the following command:

```bash
pip3 install dpctl
```

Installing the bleeding edge
------------------------

To try out the current master, install it from our
development channel on Anaconda cloud:

```bash
conda install dpctl -c dppy\label\dev
```

# Building

Refer to our [Documentation](https://intelpython.github.io/dpctl) for more information on
setting up a development environment and building `dpctl` from the source.

# Running Examples

Find our examples [here](examples).

To run these examples, use:

```bash
for script in `ls examples/python/`;
    do echo "executing ${script}";
    python examples/python/${script};
done
```

##  Cython extensions
See examples of building Cython extensions with DPC++ compiler that interoperates
with `dpctl` in the [cython folder](examples\cython).

To build these examples, run:
```bash
CC=icx CXX=dpcpp python setup.py build_ext --inplace
```
To execute extensions, refer to the `run.py` script in each folder.

# Running Tests

Tests are located [here](dpctl/tests).

To run the tests, use:
```bash
pytest --pyargs dpctl
```
