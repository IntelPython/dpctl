# Usage of dpctl Entities in Pybind11

## Description

This extension demonstrates how you can use dpctl Python types,
such as ``dpctl.SyclQueue`` and ``dpctl.program.SyclKernel``, in
Pybind11 extensions.


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
(dpctl) [17:25:27 ubuntu_vm use_dpctl_syclkernel]$ python example.py
[ 0  2  4  6  8 10 12 14 16 18 20 22 24]
```
