
# SYCL Build Pybind11 Extension

## Building 

> **NOTE:** Install scikit-build and dpcpp before next steps.

To build, run: 
```sh
python setup.py develop -- -G "Ninja" \
     -DCMAKE_C_COMPILER:PATH=icx \
     -DCMAKE_CXX_COMPILER:PATH=icpx \
     -DTBB_LIBRARY_DIR=$CONDA_PREFIX/lib \
     -DMKL_LIBRARY_DIR=${CONDA_PREFIX}/lib \
     -DMKL_INCLUDE_DIR=${CONDA_PREFIX}/include \
     -DTBB_INCLUDE_DIR=${CONDA_PREFIX}/include
```

## Running

To run the example, use:

```sh
python -m pytest tests
```

To compare Python overhead, use:

```
# build standad-alone executable
cmake --build $(find . -name cmake-build) --target standalone_cpp
# execute it
$(find . -name cmake-build)/standalone_cpp 1000 11
# launch Python computatin
python sycl_timing_solver.py 1000 11
```

Compare host times vs. C++ wall-clock times while making sure that the number of iterations is the same.
