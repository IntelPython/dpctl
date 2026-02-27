//==- _onemkl.cpp - Example of Pybind11 extension working with  -===//
//  dpctl Python objects.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Pybind11-generated extension exposing functions that
/// take dpctl Python objects, such as dpctl.SyclQueue, dpctl.SyclDevice, and
/// dpctl.memory.MemoryUSMDevice/Shared/Host as arguments.
///
//===----------------------------------------------------------------------===//

// clang-format off
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "cg_solver.hpp"
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "dpctl4pybind11.hpp"
// clang-format on

namespace py = pybind11;

using dpctl::utils::keep_args_alive;

namespace
{

void validate_usm_nbytes(const dpctl::memory::usm_memory &mem,
                         std::size_t required_nbytes,
                         const char *arg_name)
{
    const std::size_t nbytes = mem.get_nbytes();
    if (nbytes < required_nbytes) {
        throw py::value_error(std::string(arg_name) +
                              " does not have enough bytes for the requested "
                              "shape/dtype");
    }
}

void validate_positive_sizes(std::int64_t n, std::int64_t m)
{
    if (n < 0 || m < 0) {
        throw py::value_error("Dimensions must be non-negative");
    }
}

} // end anonymous namespace

std::pair<sycl::event, sycl::event>
py_gemv(sycl::queue &q,
        dpctl::memory::usm_memory matrix,
        dpctl::memory::usm_memory vector,
        dpctl::memory::usm_memory result,
        std::int64_t n,
        std::int64_t m,
        py::dtype dtype,
        std::int64_t lda,
        const std::vector<sycl::event> &depends = {})
{
    validate_positive_sizes(n, m);

    if (lda < m) {
        throw py::value_error("lda must be >= number of columns (m)");
    }

    if (!dpctl::utils::queues_are_compatible(
            q, {matrix.get_queue(), vector.get_queue(), result.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocations are not compatible with the execution queue.");
    }

    const py::ssize_t itemsize = dtype.itemsize();

    validate_usm_nbytes(matrix,
                        static_cast<std::size_t>(n) *
                            static_cast<std::size_t>(lda) * itemsize,
                        "Amatrix");
    validate_usm_nbytes(vector, static_cast<std::size_t>(m) * itemsize, "xvec");
    validate_usm_nbytes(result, static_cast<std::size_t>(n) * itemsize,
                        "resvec");

    char *mat_typeless_ptr = matrix.get_pointer();
    char *v_typeless_ptr = vector.get_pointer();
    char *r_typeless_ptr = result.get_pointer();

    sycl::event res_ev;
    const char dtype_char = dtype.char_();
    if (dtype_char == 'd') {
        using T = double;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), lda,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (dtype_char == 'f') {
        using T = float;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), lda,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (dtype_char == 'D') {
        using T = std::complex<double>;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), lda,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (dtype_char == 'F') {
        using T = std::complex<float>;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), lda,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else {
        throw std::runtime_error("Unsupported data type for gemv.");
    }

    sycl::event ht_event =
        keep_args_alive(q, {matrix, vector, result}, {res_ev});

    return std::make_pair(ht_event, res_ev);
}

template <typename T>
sycl::event sub_impl(sycl::queue q,
                     size_t n,
                     const char *v1_i,
                     const char *v2_i,
                     char *r_i,
                     const std::vector<sycl::event> &depends = {})
{
    const T *v1 = reinterpret_cast<const T *>(v1_i);
    const T *v2 = reinterpret_cast<const T *>(v2_i);
    T *r = reinterpret_cast<T *>(r_i);

    sycl::event r_ev = cg_solver::detail::sub(q, n, v1, v2, r, depends);

    return r_ev;
}

// out_r = in_v1 - in_v2
std::pair<sycl::event, sycl::event>
py_sub(sycl::queue &q,
       dpctl::memory::usm_memory in_v1,
       dpctl::memory::usm_memory in_v2,
       dpctl::memory::usm_memory out_r,
       std::int64_t n,
       py::dtype dtype,
       const std::vector<sycl::event> &depends = {})
{
    validate_positive_sizes(n, 0);

    if (!dpctl::utils::queues_are_compatible(
            q, {in_v1.get_queue(), in_v2.get_queue(), out_r.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }

    const py::ssize_t itemsize = dtype.itemsize();

    validate_usm_nbytes(in_v1, static_cast<std::size_t>(n) * itemsize, "in1");
    validate_usm_nbytes(in_v2, static_cast<std::size_t>(n) * itemsize, "in2");
    validate_usm_nbytes(out_r, static_cast<std::size_t>(n) * itemsize, "out");

    const char *in_v1_typeless_ptr = in_v1.get_pointer();
    const char *in_v2_typeless_ptr = in_v2.get_pointer();
    char *out_r_typeless_ptr = out_r.get_pointer();

    sycl::event res_ev;
    const char dtype_char = dtype.char_();
    if (dtype_char == 'd') {
        using T = double;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (dtype_char == 'f') {
        using T = float;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (dtype_char == 'D') {
        using T = std::complex<double>;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (dtype_char == 'F') {
        using T = std::complex<float>;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else {
        throw std::runtime_error("Unsupported data type for sub.");
    }

    sycl::event ht_event = keep_args_alive(q, {in_v1, in_v2, out_r}, {res_ev});

    return std::make_pair(ht_event, res_ev);
}

template <typename T> class axpy_inplace_kern;
template <typename T> class axpby_inplace_kern;

template <typename T>
sycl::event axpby_inplace_impl(sycl::queue q,
                               size_t nelems,
                               py::object pyobj_a,
                               const char *x_typeless,
                               py::object pyobj_b,
                               char *y_typeless,
                               const std::vector<sycl::event> depends = {})
{
    T a = py::cast<T>(pyobj_a);
    T b = py::cast<T>(pyobj_b);

    const T *x = reinterpret_cast<const T *>(x_typeless);
    T *y = reinterpret_cast<T *>(y_typeless);

    sycl::event res_ev =
        cg_solver::detail::axpby_inplace(q, nelems, a, x, b, y, depends);

    return res_ev;
}

// y = a * x + b * y
std::pair<sycl::event, sycl::event>
py_axpby_inplace(sycl::queue q,
                 py::object a,
                 dpctl::memory::usm_memory x,
                 py::object b,
                 dpctl::memory::usm_memory y,
                 std::int64_t n,
                 py::dtype dtype,
                 const std::vector<sycl::event> &depends = {})
{
    validate_positive_sizes(n, 0);

    if (!dpctl::utils::queues_are_compatible(q, {x.get_queue(), y.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }

    const py::ssize_t itemsize = dtype.itemsize();

    validate_usm_nbytes(x, static_cast<std::size_t>(n) * itemsize, "x");
    validate_usm_nbytes(y, static_cast<std::size_t>(n) * itemsize, "y");

    const char *x_typeless_ptr = x.get_pointer();
    char *y_typeless_ptr = y.get_pointer();

    sycl::event res_ev;
    const char dtype_char = dtype.char_();
    if (dtype_char == 'd') {
        using T = double;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (dtype_char == 'f') {
        using T = float;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (dtype_char == 'D') {
        using T = std::complex<double>;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (dtype_char == 'F') {
        using T = std::complex<float>;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else {
        throw std::runtime_error("Unsupported data type for axpby_inplace.");
    }

    sycl::event ht_event = keep_args_alive(q, {x, y}, {res_ev});

    return std::make_pair(ht_event, res_ev);
}

template <typename T>
T norm_squared_blocking_impl(sycl::queue q,
                             size_t nelems,
                             const char *r_typeless,
                             const std::vector<sycl::event> &depends = {})
{
    const T *r = reinterpret_cast<const T *>(r_typeless);

    return cg_solver::detail::norm_squared_blocking(q, nelems, r, depends);
}

template <typename T> class complex_norm_squared_blocking_kern;

template <typename T>
T complex_norm_squared_blocking_impl(
    sycl::queue q,
    size_t nelems,
    const char *r_typeless,
    const std::vector<sycl::event> &depends = {})
{
    const std::complex<T> *r =
        reinterpret_cast<const std::complex<T> *>(r_typeless);

    return cg_solver::detail::complex_norm_squared_blocking(q, nelems, r,
                                                            depends);
}

py::object py_norm_squared_blocking(sycl::queue &q,
                                    dpctl::memory::usm_memory r,
                                    std::int64_t n,
                                    py::dtype dtype,
                                    const std::vector<sycl::event> depends = {})
{
    validate_positive_sizes(n, 0);

    if (!dpctl::utils::queues_are_compatible(q, {r.get_queue()})) {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }

    const py::ssize_t itemsize = dtype.itemsize();
    validate_usm_nbytes(r, static_cast<std::size_t>(n) * itemsize, "r");

    const char *r_typeless_ptr = r.get_pointer();
    py::object res;
    const char dtype_char = dtype.char_();

    if (dtype_char == 'd') {
        using T = double;
        T n_sq = norm_squared_blocking_impl<T>(q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else if (dtype_char == 'f') {
        using T = float;
        T n_sq = norm_squared_blocking_impl<T>(q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else if (dtype_char == 'D') {
        using T = std::complex<double>;
        double n_sq = complex_norm_squared_blocking_impl<double>(
            q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else if (dtype_char == 'F') {
        using T = std::complex<float>;
        float n_sq = complex_norm_squared_blocking_impl<float>(
            q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else {
        throw std::runtime_error(
            "Unsupported data type for norm_squared_blocking.");
    }

    return res;
}

py::object py_dot_blocking(sycl::queue q,
                           dpctl::memory::usm_memory v1,
                           dpctl::memory::usm_memory v2,
                           std::int64_t n,
                           py::dtype dtype,
                           const std::vector<sycl::event> &depends = {})
{
    validate_positive_sizes(n, 0);

    if (!dpctl::utils::queues_are_compatible(q,
                                             {v1.get_queue(), v2.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }

    const py::ssize_t itemsize = dtype.itemsize();

    validate_usm_nbytes(v1, static_cast<std::size_t>(n) * itemsize, "v1");
    validate_usm_nbytes(v2, static_cast<std::size_t>(n) * itemsize, "v2");

    const char *v1_typeless_ptr = v1.get_pointer();
    const char *v2_typeless_ptr = v2.get_pointer();
    py::object res;
    const char dtype_char = dtype.char_();

    if (dtype_char == 'd') {
        using T = double;
        T *res_usm = sycl::malloc_device<T>(1, q);
        sycl::event dot_ev = oneapi::mkl::blas::row_major::dot(
            q, n, reinterpret_cast<const T *>(v1_typeless_ptr), 1,
            reinterpret_cast<const T *>(v2_typeless_ptr), 1, res_usm, depends);
        T res_v{};
        q.copy<T>(res_usm, &res_v, 1, {dot_ev}).wait_and_throw();
        sycl::free(res_usm, q);
        res = py::float_(res_v);
    }
    else if (dtype_char == 'f') {
        using T = float;
        T *res_usm = sycl::malloc_device<T>(1, q);
        sycl::event dot_ev = oneapi::mkl::blas::row_major::dot(
            q, n, reinterpret_cast<const T *>(v1_typeless_ptr), 1,
            reinterpret_cast<const T *>(v2_typeless_ptr), 1, res_usm, depends);
        T res_v(0);
        q.copy<T>(res_usm, &res_v, 1, {dot_ev}).wait_and_throw();
        sycl::free(res_usm, q);
        res = py::float_(res_v);
    }
    else if (dtype_char == 'D') {
        using T = std::complex<double>;
        T *res_usm = sycl::malloc_device<T>(1, q);
        sycl::event dotc_ev = oneapi::mkl::blas::row_major::dotc(
            q, n, reinterpret_cast<const T *>(v1_typeless_ptr), 1,
            reinterpret_cast<const T *>(v2_typeless_ptr), 1, res_usm, depends);
        T res_v{};
        q.copy<T>(res_usm, &res_v, 1, {dotc_ev}).wait_and_throw();
        sycl::free(res_usm, q);
        res = py::cast(res_v);
    }
    else if (dtype_char == 'F') {
        using T = std::complex<float>;
        T *res_usm = sycl::malloc_device<T>(1, q);
        sycl::event dotc_ev = oneapi::mkl::blas::row_major::dotc(
            q, n, reinterpret_cast<const T *>(v1_typeless_ptr), 1,
            reinterpret_cast<const T *>(v2_typeless_ptr), 1, res_usm, depends);
        T res_v{};
        q.copy<T>(res_usm, &res_v, 1, {dotc_ev}).wait_and_throw();
        sycl::free(res_usm, q);
        res = py::cast(res_v);
    }
    else {
        throw std::runtime_error("Unsupported data type for dot_blocking.");
    }

    return res;
}

int py_cg_solve(sycl::queue exec_q,
                dpctl::memory::usm_memory Amat,
                dpctl::memory::usm_memory bvec,
                dpctl::memory::usm_memory xvec,
                std::int64_t n,
                py::dtype dtype,
                double rs_tol,
                const std::vector<sycl::event> &depends = {})
{
    validate_positive_sizes(n, 0);

    if (!dpctl::utils::queues_are_compatible(
            exec_q, {Amat.get_queue(), bvec.get_queue(), xvec.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation queues are not the same as the execution queue");
    }

    const auto itemsize = dtype.itemsize();

    validate_usm_nbytes(Amat,
                        static_cast<std::size_t>(n) *
                            static_cast<std::size_t>(n) * itemsize,
                        "Amat");
    validate_usm_nbytes(bvec, static_cast<std::size_t>(n) * itemsize, "bvec");
    validate_usm_nbytes(xvec, static_cast<std::size_t>(n) * itemsize, "xvec");

    const char *A_ch = Amat.get_pointer();
    const char *b_ch = bvec.get_pointer();
    char *x_ch = xvec.get_pointer();
    const char dtype_char = dtype.char_();

    if (dtype_char == 'd') {
        using T = double;
        int iters = cg_solver::cg_solve<T>(
            exec_q, n, reinterpret_cast<const T *>(A_ch),
            reinterpret_cast<const T *>(b_ch), reinterpret_cast<T *>(x_ch),
            depends, static_cast<T>(rs_tol));

        return iters;
    }
    else if (dtype_char == 'f') {
        using T = float;
        int iters = cg_solver::cg_solve<T>(
            exec_q, n, reinterpret_cast<const T *>(A_ch),
            reinterpret_cast<const T *>(b_ch), reinterpret_cast<T *>(x_ch),
            depends, static_cast<T>(rs_tol));

        return iters;
    }
    else {
        throw std::runtime_error("Unsupported data type for cg_solve. Use "
                                 "single or double precision.");
    }
}

PYBIND11_MODULE(_onemkl, m)
{
    m.def("gemv", &py_gemv, "Uses oneMKL to compute dot(matrix, vector)",
          py::arg("exec_queue"), py::arg("Amatrix"), py::arg("xvec"),
          py::arg("resvec"), py::arg("nrows"), py::arg("ncols"),
          py::arg("dtype"), py::arg("lda") = -1,
          py::arg("depends") = py::list());
    m.def("sub", &py_sub, "Subtraction: out = v1 - v2", py::arg("exec_queue"),
          py::arg("in1"), py::arg("in2"), py::arg("out"), py::arg("nelems"),
          py::arg("dtype"), py::arg("depends") = py::list());
    m.def("axpby_inplace", &py_axpby_inplace, "y = a * x + b * y",
          py::arg("exec_queue"), py::arg("a"), py::arg("x"), py::arg("b"),
          py::arg("y"), py::arg("nelems"), py::arg("dtype"),
          py::arg("depends") = py::list());
    m.def("norm_squared_blocking", &py_norm_squared_blocking, "norm(r)**2",
          py::arg("exec_queue"), py::arg("r"), py::arg("nelems"),
          py::arg("dtype"), py::arg("depends") = py::list());
    m.def("dot_blocking", &py_dot_blocking, "<v1, v2>", py::arg("exec_queue"),
          py::arg("v1"), py::arg("v2"), py::arg("nelems"), py::arg("dtype"),
          py::arg("depends") = py::list());

    m.def("cpp_cg_solve", &py_cg_solve,
          "Dispatch to call C++ implementation of cg_solve",
          py::arg("exec_queue"), py::arg("Amat"), py::arg("bvec"),
          py::arg("xvec"), py::arg("n"), py::arg("dtype"),
          py::arg("rs_squared_tolerance") = py::float_(1e-20),
          py::arg("depends") = py::list());
}
