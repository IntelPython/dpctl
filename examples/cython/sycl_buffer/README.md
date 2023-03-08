# SYCL Extension Working NumPy Array Input via SYCL Buffers

## Decription

Cython function expecting a 2D array in a C-contiguous layout that
computes column-wise total by using SYCL oneMKL (as GEMV call with
an all-units vector).

The example illustrates compiling SYCL extension that is linking to
oneMKL.


## Compiling

> **NOTE:** Make sure oneAPI is activated, $ONEAPI_ROOT must be set.

To compile the example, run:
```
python setup.py develop
```

## Running

```
(dev_dpctl) opavlyk@opavlyk-mobl:~/repos/dpctl/examples/cython/sycl_buffer$ ipython
Python 3.9.12 (main, Jun  1 2022, 11:38:51)
Type 'copyright', 'credits' or 'license' for more information
IPython 8.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import syclbuffer as sb, numpy as np, dpctl

In [2]: x = np.random.randn(10**6, 7).astype(np.float32)

In [3]: sb.columnwise_total(x)
Out[3]:
array([ -810.02496 ,    42.692146,  -786.71075 , -1417.643   ,
       -1096.2424  ,   212.33067 ,    18.40631 ], dtype=float32)

In [4]: np.sum(x, axis=0)
Out[4]:
array([ -810.03296 ,    42.68893 ,  -786.7023  , -1417.648   ,
       -1096.2699  ,   212.32564 ,    18.412518], dtype=float32)
```

### Running bench.py:

```
$ python scripts/bench.py
```

### Running tests:

```
$ python -m pytest tests
```
