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
2. Build conda package
```bash
conda build conda-recipe
```
On Windows to cope with [long file names](https://github.com/IntelPython/dpctl/issues/15):
```cmd
conda build --croot=C:/tmp conda-recipe
```

:warning: **You could face issues with conda-build=3.20**: Use conda-build=3.18!

3. Install conda package
```bash
conda install dpctl
```

Using dpCtl
===========
dpCtl relies on DPC++ runtime. With Intel oneAPI installed you should activate it.

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
