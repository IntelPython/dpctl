
# SYCL Build Pybind11 Extension

## Building

> **NOTE:** Install scikit-build and dpcpp before next steps.

To build on Linux, run:
```bash
python -m pip install -e . \
     -Ccmake.define.CMAKE_C_COMPILER:PATH=icx \
     -Ccmake.define.CMAKE_CXX_COMPILER:PATH=icpx \
     -Ccmake.define.TBB_LIBRARY_DIR=$CONDA_PREFIX/lib \
     -Ccmake.define.MKL_LIBRARY_DIR=${CONDA_PREFIX}/lib \
     -Ccmake.define.MKL_INCLUDE_DIR=${CONDA_PREFIX}/include \
     -Ccmake.define.TBB_INCLUDE_DIR=${CONDA_PREFIX}/include
```

To build on Windows, run:
```bash
python -m pip install -e . \
     -Ccmake.define.CMAKE_C_COMPILER:PATH=icx \
     -Ccmake.define.CMAKE_CXX_COMPILER:PATH=icx \
     -Ccmake.define.TBB_LIBRARY_DIR=$CONDA_PREFIX/lib \
     -Ccmake.define.MKL_LIBRARY_DIR=${CONDA_PREFIX}/lib \
     -Ccmake.define.MKL_INCLUDE_DIR=${CONDA_PREFIX}/include \
     -Ccmake.define.TBB_INCLUDE_DIR=${CONDA_PREFIX}/include
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
