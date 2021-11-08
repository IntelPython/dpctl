[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Coverage Status](https://coveralls.io/repos/github/IntelPython/dpctl/badge.svg?branch=master)](https://coveralls.io/github/IntelPython/dpctl?branch=master)

About dpctl
===========

<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo" width="75"/>

Data Parallel Control (`dpctl`) is a Python library that allows a user to *control* the execution placement of 
a [compute kernel](https://en.wikipedia.org/wiki/Compute_kernel) on an [XPU](https://www.intel.com/content/www/us/en/newsroom/news/xpu-vision-oneapi-server-gpu.html).
The compute kernel can be either a code written by the user, *e.g.*, using `numba-dppy`, or part of a library like oneMKL.
The `dpctl` library is built upon the [SYCL standard](https://www.khronos.org/sycl/) and implements Python bindings for a subset of the
standard [runtime classes](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_sycl_runtime_classes)
that allow users to query platforms, discover and represent devices and sub-devices, and construct contexts and queues. 
In addition, `dpctl` features classes for [SYCL Unified Shared Memory (USM)](https://link.springer.com/chapter/10.1007/978-1-4842-5574-2_6)
management and implements a tensor [array API](https://data-apis.org/array-api/latest/). 

The library also assists authors of Python native extensions written in C, Cython, or pybind11 to access `dpctl` objects 
representing SYCL devices, queues, memory, and tensors.

`Dpctl` is the core part of a larger family of 
[data-parallel Python libraries and tools](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) 
to program XPUs. The library is available via [conda](https://anaconda.org/intel/dpctl) and [pip](https://pypi.org/project/dpctl/).
It also comes with [Intel(R) Distribution for Python*](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-python.html) (IDP).

Installing `dpctl`
==================

Using Intel oneAPI
------------------

`dpctl` is packaged as part of the quarterly Intel oneAPI releases. To get the library from the latest
oneAPI release please follow the instructions from Intel's
[oneAPI installation guide](https://www.intel.com/content/www/us/en/developer/articles/guide/installation-guide-for-oneapi-toolkits.html).
Note that you will need to install the Intel BaseKit toolkit to get IDP and `dpctl`.

Using Conda
-----------

`dpctl` packages are available on the Intel channel on Annaconda cloud. You an use the following to install `dpctl` from there:

```bash
conda install dpctl -c intel
```

Using PyPi
----------

`dpctl` is also available from PyPi and to install run:

```bash
pip3 install dpctl
```

Installing bleeding edge
------------------------

If you want to try out the current master, you can install it from our development channel on Annocaonda cloud:

```bash
conda install dpctl -c dppy\label\dev
```

Building `dpctl`
================

Please refer our [getting started user guide](https://intelpython.github.io/dpctl) for more 
information on setting up a development environment and building `dpctl` from source.

Using dpctl
===========
dpctl requires a DPC++ runtime to be installed. It is suggested to use install the `dpcpp_cpp_rt` package from conda.
Alternatively, a full Intel oneAPI installation may also be used. You will have to activate your
oneAPI installation by using the following commands:

On Windows:
```cmd
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
```
On Linux:
```bash
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
```
Examples
========
See examples in folder `examples`.

Run python examples:
```bash
for script in `ls examples/python/`; do echo "executing ${script}"; python examples/python/${script}; done
```

Examples of building Cython extensions with DPC++ compiler, that interoperate
with dpctl can be found in folder `cython`.

Each example in `cython` folder can be built using
`CC=clang CXX=dpcpp python setup.py build_ext --inplace`.
Please refer to `run.py` script in respective folders to execute extensions.

Tests
=====
See tests in folder `dpctl/tests`.

Run tests:
```bash
pytest --pyargs dpctl
```
