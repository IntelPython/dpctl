Example of SYCL built pybind11 extension

To build, use (assumes scikit-build and dpcpp) is installed

```sh
python setup.py develop -- -G "Ninja" -DCMAKE_C_COMPILER:PATH=icx -DCMAKE_CXX_COMPILER:PATH=icpx -DTBB_LIBRARY_DIR=$CONDA_PREFIX/lib -DMKL_LIBRARY_DIR=${CONDA_PREFIX}/lib -DMKL_INCLUDE_DIR=${CONDA_PREFIX}/include -DTBB_INCLUDE_DIR=${CONDA_PREFIX}/include
```

To run test suite

```sh
python -m pytest tests
```
