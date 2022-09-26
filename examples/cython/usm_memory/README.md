# Example of working with USM memory

## Description

## Building

Make sure oneAPI is activated. Environment variable `$ONEAPI_ROOT` must be set.


```
$ CC=icx CXX=dpcpp LD_SHARED="dpcpp -shared" \
  CXXFLAGS=-fno-sycl-early-optimizations python setup.py build_ext --inplace
```

## Running

```
$ python run.py
```

which gives sample output:

```
True
Using : Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz
Elapsed: 0.9255791641771793
Using : Intel(R) Gen9
Elapsed: 0.32811625860631466
```
