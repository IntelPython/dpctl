#pragma once

#pragma once
#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "utils/type_dispatch.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{
namespace type_utils
{

/*! @brief Produce dtype from a type number */
extern py::dtype _dtype_from_typenum(td_ns::typenum_t);

/*! @brief Lookup typeid of the result from typeid of
 *         argument and the mapping table */
extern int _result_typeid(int, const int *);

} // namespace type_utils
} // namespace py_internal
} // namespace tensor
} // namespace dpctl
