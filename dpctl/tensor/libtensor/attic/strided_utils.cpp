#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "utils/strided_iters.hpp"
#include <vector>

namespace py = pybind11;

py::object contract_iter(const std::vector<size_t> &shape,
                         const std::vector<Py_ssize_t> &strides)
{
    int n = shape.size();
    if (n != static_cast<int>(strides.size()))
        throw std::runtime_error("Unequal lengths");
    Py_ssize_t disp = 0;

    std::vector<size_t> shape_vec = shape;
    std::vector<Py_ssize_t> strides_vec = strides;
    int new_n = simplify_iteration_stride<size_t, Py_ssize_t>(
        n, shape_vec.data(), strides_vec.data(), disp);
    shape_vec.resize(new_n);
    strides_vec.resize(new_n);
    py::list new_shape = py::cast(shape_vec);
    py::list new_strides = py::cast(strides_vec);
    py::object p = py::cast(disp);

    return py::make_tuple(new_shape, new_strides, p);
}

PYBIND11_MODULE(strided_utils, m)
{
    m.def("contract_iter", &contract_iter);
}
