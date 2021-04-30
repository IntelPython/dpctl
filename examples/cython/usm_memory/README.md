#1 Example of working with USM memory

#2 Description

#2 Building

Make sure oneAPI is activated. Environment variable `$ONEAPI_ROOT` must be set.


```
$ CC=clang CXX=dpcpp LD_SHARED="dpcpp -shared" \
  CXXFLAGS=-fno-sycl-early-optimizations python setup.py build_ext --inplace
```

#2 Running

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
