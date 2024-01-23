#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

extern void init_dot(py::module_ m);

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
