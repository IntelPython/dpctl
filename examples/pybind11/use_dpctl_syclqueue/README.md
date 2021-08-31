# Usage of dpctl entities in Pybind11

This extension demonstrates how dpctl Python types,
such as dpctl.SyclQueue could be used in Pybind11
extensions.


# Building extension

```
source /opt/intel/oneapi/compiler/latest/env/vars.sh
CXX=dpcpp CC=dpcpp python setup.py build_ext --inplace
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
