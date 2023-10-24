#include "dpctl4pybind11.hpp"
#include <CL/sycl.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "elementwise_functions_type_utils.hpp"
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

py::dtype _dtype_from_typenum(td_ns::typenum_t dst_typenum_t)
{
    switch (dst_typenum_t) {
    case td_ns::typenum_t::BOOL:
        return py::dtype("?");
    case td_ns::typenum_t::INT8:
        return py::dtype("i1");
    case td_ns::typenum_t::UINT8:
        return py::dtype("u1");
    case td_ns::typenum_t::INT16:
        return py::dtype("i2");
    case td_ns::typenum_t::UINT16:
        return py::dtype("u2");
    case td_ns::typenum_t::INT32:
        return py::dtype("i4");
    case td_ns::typenum_t::UINT32:
        return py::dtype("u4");
    case td_ns::typenum_t::INT64:
        return py::dtype("i8");
    case td_ns::typenum_t::UINT64:
        return py::dtype("u8");
    case td_ns::typenum_t::HALF:
        return py::dtype("f2");
    case td_ns::typenum_t::FLOAT:
        return py::dtype("f4");
    case td_ns::typenum_t::DOUBLE:
        return py::dtype("f8");
    case td_ns::typenum_t::CFLOAT:
        return py::dtype("c8");
    case td_ns::typenum_t::CDOUBLE:
        return py::dtype("c16");
    default:
        throw py::value_error("Unrecognized dst_typeid");
    }
}

int _result_typeid(int arg_typeid, const int *fn_output_id)
{
    if (arg_typeid < 0 || arg_typeid >= td_ns::num_types) {
        throw py::value_error("Input typeid " + std::to_string(arg_typeid) +
                              " is outside of expected bounds.");
    }

    return fn_output_id[arg_typeid];
}

} // namespace type_utils
} // namespace py_internal
} // namespace tensor
} // namespace dpctl
