# SYCL Extension Working NumPy Array Input via SYCL Buffers

## Decription

Cython* function expecting a 2D array in a C-contiguous layout that
computes column-wise total by using SYCL oneMKL (as GEMV call with
an all-units vector).

The example illustrates compiling SYCL* extension linking to oneMKL.


## Compiling

> **NOTE:** Make sure oneAPI is activated, $ONEAPI_ROOT must be set.

To compile the example, run:
```
CC=icx CXX=dpcpp python setup.py build_ext --inplace
```

## Running

```
# SYCL_DEVICE_FILTER=opencl sets SYCL backend to OpenCL to avoid a
# transient issue with MKL's using the default Level-0 backend
(idp) [08:16:12 ansatnuc04 simple]$ SYCL_DEVICE_FILTER=opencl ipython
Python 3.7.7 (default, Jul 14 2020, 22:02:37)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.17.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import syclbuffer as sb, numpy as np, dpctl

In [2]: x = np.random.randn(10**4, 2500)

In [3]: %time m1 = np.sum(x, axis=0)
CPU times: user 22.3 ms, sys: 160 µs, total: 22.5 ms
Wall time: 21.2 ms

In [4]: %time m = sb.columnwise_total(x)  # first time is slower, due to JIT overhead
CPU times: user 207 ms, sys: 36.1 ms, total: 243 ms
Wall time: 248 ms

In [5]: %time m = sb.columnwise_total(x)
CPU times: user 8.89 ms, sys: 4.12 ms, total: 13 ms
Wall time: 12.4 ms

In [6]: %time m = sb.columnwise_total(x)
CPU times: user 4.82 ms, sys: 8.06 ms, total: 12.9 ms
Wall time: 12.3 ms
```

### Running bench.py:

```
========== Executing warm-up ==========
NumPy result:  [1. 1. 1. ... 1. 1. 1.]
SYCL(Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz) result: [1. 1. 1. ... 1. 1. 1.]
SYCL(Intel(R) Gen9 HD Graphics NEO) result: [1. 1. 1. ... 1. 1. 1.]
Times for 'opencl:cpu:0'
[2.864787499012891, 2.690436460019555, 2.5902308400254697, 2.5802528870408423, 2.538990616973024]
Times for 'opencl:gpu:0'
[1.9769684099592268, 2.3491444009705447, 2.293720397981815, 2.391633405990433, 1.9465659779962152]
Times for NumPy
[3.4011058019823395, 3.07286038500024, 3.0390414349967614, 3.0305576199898496, 3.002687797998078]
```

### Running run.py:

```
(idp) [09:14:53 ansatnuc04 sycl_buffer]$ SYCL_DEVICE_FILTER=opencl python run.py
Result computed by NumPy
[  0.27170187 -23.36798583   7.31326489  -1.95121928]
Result computed by SYCL extension
[  0.27170187 -23.36798583   7.31326489  -1.95121928]

Running on:  Intel(R) Gen9 HD Graphics NEO
[  0.27170187 -23.36798583   7.31326489  -1.95121928]
Running on:  Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz
[  0.27170187 -23.36798583   7.31326489  -1.95121928]
```
