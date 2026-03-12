# Usage of dpctl Entities in Pybind11

## Description

This extension demonstrates how you can use dpctl Python types,
such as ``dpctl.SyclQueue``, in Pybind11
extensions.


## Building

To build the extension, run:
```
source /opt/intel/oneapi/compiler/latest/env/vars.sh
CXX=icpx python -m pip install .
python -m pytest tests
python example.py
```

# Sample output

```
$ python example.py
EU count returned by Pybind11 extension 96
EU count computed by dpctl 96
Device's global memory size:  7445078016 bytes
Device's local memory size:  65536 bytes

Computing modular reduction using SYCL on a NumPy array
Offloaded result agrees with reference one computed by NumPy
```
