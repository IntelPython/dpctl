# Working with USM Memory


## Building

> **NOTE:** Make sure oneAPI is activated, $ONEAPI_ROOT must be set.

To build the example, run:
```
$ CC=icx CXX=dpcpp LD_SHARED="dpcpp -shared" \
  CXXFLAGS=-fno-sycl-early-optimizations python setup.py build_ext --inplace
```

## Running

```
$ python run.py
```

It gives the example output:

```
True
Using : Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz
Elapsed: 0.9255791641771793
Using : Intel(R) Gen9
Elapsed: 0.32811625860631466
```
