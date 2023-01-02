# Usage of dpctl Entities in Pybind11

## Description

This extension demonstrates how you can use dpctl Python types,
such as ``dpctl.SyclQueue``, in Pybind11
extensions.


## Building

To build the extension, run:
```
source /opt/intel/oneapi/compiler/latest/env/vars.sh
CXX=icpx python setup.py build_ext --inplace
python -m pytest tests
python example.py
```

# Sample output

```
(idp) [17:25:27 ansatnuc04 use_dpctl_syclqueue]$ python example.py
EU count returned by Pybind11 extension 24
EU count computed by dpctl 24

Computing modular reduction using SYCL on a NumPy array
Offloaded result agrees with reference one computed by NumPy
```
