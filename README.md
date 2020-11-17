[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

What?
====
A lightweight Python package exposing a subset of SYCL functionalities.

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
2. Activate oneAPI DPC++ compiler environmnet and build conda package
```bash
export ONEAPI_ROOT=/opt/intel/oneapi
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
conda build conda-recipe
```
On Windows to cope with [long file names](https://github.com/IntelPython/dpctl/issues/15)
use `croot` with short folder path:
```cmd
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI\"
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
conda build --croot=C:/tmp conda-recipe
```

:warning: **You could face issues with conda-build=3.20**: Use conda-build=3.18!

3. Install conda package
```bash
conda install dpctl
```

Using dpCtl
===========
dpCtl relies on DPC++ runtime. With Intel oneAPI installed you could activate it.

On Windows:
```cmd
call "%ONEAPI_ROOT%\compiler\latest\env\vars.bat"
```
On Linux:
```bash
source ${ONEAPI_ROOT}/compiler/latest/env/vars.sh
```

When dpCtl is installed via conda package
then it uses DPC++ runtime from `dpcpp_cpp_rt` package
and it is not necessary to activate oneAPI DPC++ compiler environment.

`dpcpp_cpp_rt` package is provided by Intel Python from oneAPI.

Examples
========
See examples in folder `examples`.

Run examples:
```bash
python examples/create_sycl_queues.py
```

Tests
=====
See tests in folder `dpctl/tests`.

Run tests:
```bash
python -m unittest dpctl.tests
```
