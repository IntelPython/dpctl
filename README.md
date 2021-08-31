[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Coverage Status](https://coveralls.io/repos/github/IntelPython/dpctl/badge.svg?branch=master)](https://coveralls.io/github/IntelPython/dpctl?branch=master)

What?
====
A lightweight Python package exposing a subset of SYCL functionalities, part of [![](https://spec.oneapi.io/oneapi-logo-white-scaled.jpg)](https://oneapi.io).

Requirements
============
- Install Conda
- Install Intel oneAPI
    - Set environment variable `ONEAPI_ROOT`
        - Windows: `C:\Program Files (x86)\Intel\oneAPI\`
        - Linux: `/opt/intel/oneapi`
- Install OpenCL HD graphics drivers

Build and Install Conda Package
==================================
1. Create and activate conda build environment
```bash
conda create -n build-env conda-build
conda activate build-env
```
2. Set environment variable `ONEAPI_ROOT` and build conda package
```bash
export ONEAPI_ROOT=/opt/intel/oneapi
conda build conda-recipe -c ${ONEAPI_ROOT}/conda_channel
```
On Windows to cope with [long file names](https://github.com/IntelPython/dpctl/issues/15)
use `croot` with short folder path:
```cmd
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\"
conda build --croot=C:/tmp conda-recipe -c "%ONEAPI_ROOT%\conda_channel"
```

:warning: **You could face issues with conda-build=3.20**: Use conda-build=3.18!

3. Install conda package
```bash
conda install dpctl
```

Build and Install with setuptools
=================================
dpctl relies on DPC++ runtime. With Intel oneAPI installed you should activate it.
`setup.py` requires environment variable `ONEAPI_ROOT` and following packages
installed:
- `cython`
- `numpy`
- `cmake` - for building C API
- `ninja` - only on Windows

You need DPC++ to build dpctl. If you want to build using the DPC++ in a
oneAPI distribution, activate DPC++ compiler as follows:
```bash
export ONEAPI_ROOT=/opt/intel/oneapi
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
```

For install:
```cmd
python setup.py install
```

For development:
```cmd
python setup.py develop
```

It is also possible to build dpctl using [DPC++ toolchain](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) instead of oneAPI DPC++. Instead of activating the oneAPI environment, indicate the toolchain installation prefix with `--sycl-compiler-prefix` option, e.g.

```cmd
python setup.py develop --sycl-compiler-prefix=${DPCPP_ROOT}/llvm/build
```

Please use `python setup.py develop --help` for more details.

Install Wheel Package from Pypi
==================================
1. Install dpctl
```cmd
python -m pip install --index-url https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple dpctl
```
Note: dpctl wheel package is placed on Pypi, but some of its dependencies (like Intel numpy) are in Anaconda Cloud.
That is why install command requires additional intel Pypi channel from Anaconda Cloud.

2. Set path to Performance Libraries in case of using venv or system Python:
On Linux:
```cmd
export LD_LIBRARY_PATH=<path_to_your_env>/lib
```
On Windows:
```cmd
set PATH=<path_to_your_env>\bin;<path_to_your_env>\Library\bin;%PATH%
```

Using dpctl
===========
dpctl relies on DPC++ runtime. With Intel oneAPI installed you could activate it.

On Windows:
```cmd
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
```
On Linux:
```bash
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
```

When dpctl is installed via conda package
then it uses DPC++ runtime from `dpcpp_cpp_rt` package
and it is not necessary to activate oneAPI DPC++ compiler environment.

`dpcpp_cpp_rt` package is provided by oneAPI `conda_channel`.

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
