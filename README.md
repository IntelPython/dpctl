What?
====
A lightweight Python package exposing a subset of OpenCL and SYCL
functionalities.

Requirements
============
- Install Conda
- Install Intel oneAPI
    - Set environment variable `ONEAPI_ROOT`
        - Windows: `C:\Program Files (x86)\Intel\oneAPI\`
        - Linux: `/opt/intel/oneapi`
- Install OpenCL HD graphics drivers

Building and Install Conda Package
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
On Windows to cope with [long file names](https://github.com/IntelPython/pydppl/issues/15):
```cmd
conda build --croot=C:/tmp conda-recipe
```
3. Install conda package
```bash
conda install pydppl
```

Using PyDPPL
============
PyDPPL relies on SYCL runtime. With Intel oneAPI installed you should activate it.

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
