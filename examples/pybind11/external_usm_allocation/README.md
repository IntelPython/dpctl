# Exposing USM Allocations Made by the Native Code to dpctl

This extension demonstrates how a Python object backed by
a native class, which allocates USM memory, can expose it
to the `dpctl.memory` entities using `__sycl_usm_array_interface__`.


## Building

To build the example, run:

```
source /opt/intel/oneapi/compiler/latest/env/vars.sh
CXX=dpcpp CC=dpcpp python setup.py build_ext --inplace
python -m pytest tests
python example.py
```

## Example output

```
(idp) [12:43:20 ansatnuc04 external_usm_allocation]$ python example.py
<external_usm_allocation._external_usm_alloc.DMatrix object at 0x7f2b98b4cef0>
{'data': [94846745444352, True], 'shape': (5, 5), 'strides': None, 'version': 1, 'typestr': '|f8', 'syclobj': <capsule object "SyclQueueRef" at 0x7f2b9b941d80>}
shared

[1.0, 1.0, 1.0, 2.0, 2.0]
[1.0, 0.0, 1.0, 2.0, 2.0]
[1.0, 1.0, 0.0, 2.0, 2.0]
[0.0, 0.0, 0.0, 3.0, -1.0]
[0.0, 0.0, 0.0, -1.0, 5.0]
```
