//==- _onemkl.cpp - Example of Pybind11 extension working with  -===//
//  dpctl Python objects.
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
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
/// dpctl.tensor.usm_ndarray as arguments.
///
//===----------------------------------------------------------------------===//

// clang-format off
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "cg_solver.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "dpctl4pybind11.hpp"
// clang-format on

namespace py = pybind11;

using dpctl::utils::keep_args_alive;

std::pair<sycl::event, sycl::event>
py_gemv(sycl::queue q,
        dpctl::tensor::usm_ndarray matrix,
        dpctl::tensor::usm_ndarray vector,
        dpctl::tensor::usm_ndarray result,
        const std::vector<sycl::event> &depends = {})
{
    if (matrix.get_ndim() != 2 || vector.get_ndim() != 1 ||
        result.get_ndim() != 1) {
        throw std::runtime_error(
            "Inconsistent dimensions, expecting matrix and a vector");
    }

    py::ssize_t n = matrix.get_shape(0); // get 0-th element of the shape
    py::ssize_t m = matrix.get_shape(1);

    py::ssize_t v_dim = vector.get_shape(0);
    py::ssize_t r_dim = result.get_shape(0);
    if (v_dim != m || r_dim != n) {
        throw std::runtime_error("Inconsistent shapes.");
    }

    if (!dpctl::utils::queues_are_compatible(
            q, {matrix.get_queue(), vector.get_queue(), result.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocations are not compatible with the execution queue.");
    }

    auto &api = dpctl::detail::dpctl_capi::get();

    if (!((matrix.is_c_contiguous()) &&
          (vector.is_c_contiguous() || vector.is_f_contiguous()) &&
          (result.is_c_contiguous() || result.is_f_contiguous())))
    {
        throw std::runtime_error("Arrays must be contiguous.");
    }

    int mat_typenum = matrix.get_typenum();
    int v_typenum = vector.get_typenum();
    int r_typenum = result.get_typenum();

    if ((mat_typenum != v_typenum) || (r_typenum != v_typenum) ||
        !((v_typenum == api.UAR_DOUBLE_) || (v_typenum == api.UAR_FLOAT_) ||
          (v_typenum == api.UAR_CDOUBLE_) || (v_typenum == api.UAR_CFLOAT_)))
    {
        std::cout << "Found: [" << mat_typenum << ", " << v_typenum << ", "
                  << r_typenum << "]" << std::endl;
        std::cout << "Expected: [" << UAR_DOUBLE << ", " << UAR_FLOAT << ", "
                  << UAR_CDOUBLE << ", " << UAR_CFLOAT << "]" << std::endl;
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    char *mat_typeless_ptr = matrix.get_data();
    char *v_typeless_ptr = vector.get_data();
    char *r_typeless_ptr = result.get_data();

    sycl::event res_ev;
    if (v_typenum == api.UAR_DOUBLE_) {
        using T = double;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (v_typenum == api.UAR_FLOAT_) {
        using T = float;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (v_typenum == api.UAR_CDOUBLE_) {
        using T = std::complex<double>;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else if (v_typenum == api.UAR_CFLOAT_) {
        using T = std::complex<float>;
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            q, oneapi::mkl::transpose::nontrans, n, m, T(1),
            reinterpret_cast<T *>(mat_typeless_ptr), m,
            reinterpret_cast<T *>(v_typeless_ptr), 1, T(0),
            reinterpret_cast<T *>(r_typeless_ptr), 1, depends);
        res_ev = gemv_ev;
    }
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
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
py_sub(sycl::queue q,
       dpctl::tensor::usm_ndarray in_v1,
       dpctl::tensor::usm_ndarray in_v2,
       dpctl::tensor::usm_ndarray out_r,
       const std::vector<sycl::event> &depends = {})
{
    if (in_v1.get_ndim() != 1 || in_v2.get_ndim() != 1 || out_r.get_ndim() != 1)
    {
        throw std::runtime_error("Inconsistent dimensions, expecting vectors");
    }

    py::ssize_t n = in_v1.get_shape(0); // get length of the vector

    if (n != in_v2.get_shape(0) || n != out_r.get_shape(0)) {
        throw std::runtime_error("Vectors must have the same length");
    }

    if (!dpctl::utils::queues_are_compatible(
            q, {in_v1.get_queue(), in_v2.get_queue(), out_r.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }

    auto &api = dpctl::detail::dpctl_capi::get();

    if (!((in_v1.is_c_contiguous() || in_v1.is_f_contiguous()) &&
          (in_v2.is_c_contiguous() || in_v2.is_f_contiguous()) &&
          (out_r.is_c_contiguous() || out_r.is_f_contiguous())))
    {
        throw std::runtime_error("Vectors must be contiguous.");
    }

    int in_v1_typenum = in_v1.get_typenum();
    int in_v2_typenum = in_v2.get_typenum();
    int out_r_typenum = out_r.get_typenum();

    if ((in_v2_typenum != in_v1_typenum) || (out_r_typenum != in_v1_typenum) ||
        !((in_v1_typenum == api.UAR_DOUBLE_) ||
          (in_v1_typenum == api.UAR_FLOAT_) ||
          (in_v1_typenum == api.UAR_CDOUBLE_) ||
          (in_v1_typenum == api.UAR_CFLOAT_)))
    {
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    const char *in_v1_typeless_ptr = in_v1.get_data();
    const char *in_v2_typeless_ptr = in_v2.get_data();
    char *out_r_typeless_ptr = out_r.get_data();

    sycl::event res_ev;
    if (out_r_typenum == api.UAR_DOUBLE_) {
        using T = double;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (out_r_typenum == api.UAR_FLOAT_) {
        using T = float;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (out_r_typenum == api.UAR_CDOUBLE_) {
        using T = std::complex<double>;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else if (out_r_typenum == api.UAR_CFLOAT_) {
        using T = std::complex<float>;
        res_ev = sub_impl<T>(q, n, in_v1_typeless_ptr, in_v2_typeless_ptr,
                             out_r_typeless_ptr, depends);
    }
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
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
                 dpctl::tensor::usm_ndarray x,
                 py::object b,
                 dpctl::tensor::usm_ndarray y,
                 const std::vector<sycl::event> &depends = {})
{

    if (x.get_ndim() != 1 || y.get_ndim() != 1) {
        throw std::runtime_error("Inconsistent dimensions, expecting vectors");
    }

    py::ssize_t n = x.get_shape(0); // get length of the vector

    if (n != y.get_shape(0)) {
        throw std::runtime_error("Vectors must have the same length");
    }

    if (!dpctl::utils::queues_are_compatible(q, {x.get_queue(), y.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }
    auto &api = dpctl::detail::dpctl_capi::get();

    if (!((x.is_c_contiguous() || x.is_f_contiguous()) &&
          (y.is_c_contiguous() || y.is_f_contiguous())))
    {
        throw std::runtime_error("Vectors must be contiguous.");
    }

    int x_typenum = x.get_typenum();
    int y_typenum = y.get_typenum();

    if ((x_typenum != y_typenum) ||
        !((x_typenum == api.UAR_DOUBLE_) || (x_typenum == api.UAR_FLOAT_) ||
          (x_typenum == api.UAR_CDOUBLE_) || (x_typenum == api.UAR_CFLOAT_)))
    {
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    const char *x_typeless_ptr = x.get_data();
    char *y_typeless_ptr = y.get_data();

    sycl::event res_ev;
    if (x_typenum == api.UAR_DOUBLE_) {
        using T = double;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (x_typenum == api.UAR_FLOAT_) {
        using T = float;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (x_typenum == api.UAR_CDOUBLE_) {
        using T = std::complex<double>;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else if (x_typenum == api.UAR_CFLOAT_) {
        using T = std::complex<float>;
        res_ev = axpby_inplace_impl<T>(q, n, a, x_typeless_ptr, b,
                                       y_typeless_ptr, depends);
    }
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
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

py::object py_norm_squared_blocking(sycl::queue q,
                                    dpctl::tensor::usm_ndarray r,
                                    const std::vector<sycl::event> depends = {})
{
    if (r.get_ndim() != 1) {
        throw std::runtime_error("Expecting a vector");
    }

    py::ssize_t n = r.get_shape(0); // get length of the vector

    int r_flags = r.get_flags();

    if (!(r.is_c_contiguous() || r.is_f_contiguous())) {
        throw std::runtime_error("Vector must be contiguous.");
    }

    if (!dpctl::utils::queues_are_compatible(q, {r.get_queue()})) {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }

    auto &api = dpctl::detail::dpctl_capi::get();

    int r_typenum = r.get_typenum();
    if ((r_typenum != api.UAR_DOUBLE_) && (r_typenum != api.UAR_FLOAT_) &&
        (r_typenum != api.UAR_CDOUBLE_) && (r_typenum != api.UAR_CFLOAT_))
    {
        throw std::runtime_error(
            "Only real and complex floating point arrays are supported.");
    }

    const char *r_typeless_ptr = r.get_data();
    py::object res;

    if (r_typenum == api.UAR_DOUBLE_) {
        using T = double;
        T n_sq = norm_squared_blocking_impl<T>(q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else if (r_typenum == api.UAR_FLOAT_) {
        using T = float;
        T n_sq = norm_squared_blocking_impl<T>(q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else if (r_typenum == api.UAR_CDOUBLE_) {
        using T = std::complex<double>;
        double n_sq = complex_norm_squared_blocking_impl<double>(
            q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else if (r_typenum == api.UAR_CFLOAT_) {
        using T = std::complex<float>;
        float n_sq = complex_norm_squared_blocking_impl<float>(
            q, n, r_typeless_ptr, depends);
        res = py::float_(n_sq);
    }
    else {
        throw std::runtime_error("Type dispatch ran into trouble.");
    }

    return res;
}

py::object py_dot_blocking(sycl::queue q,
                           dpctl::tensor::usm_ndarray v1,
                           dpctl::tensor::usm_ndarray v2,
                           const std::vector<sycl::event> &depends = {})
{
    if (v1.get_ndim() != 1 || v2.get_ndim() != 1) {
        throw std::runtime_error("Expecting two vectors");
    }

    py::ssize_t n = v1.get_shape(0); // get length of the vector

    if (n != v2.get_shape(0)) {
        throw std::runtime_error("Length of vectors are not the same");
    }

    if (!(v1.is_c_contiguous() || v1.is_f_contiguous()) ||
        !(v2.is_c_contiguous() || v2.is_f_contiguous()))
    {
        throw std::runtime_error("Vectors must be contiguous.");
    }

    if (!dpctl::utils::queues_are_compatible(q,
                                             {v1.get_queue(), v2.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation is not bound to the context in execution queue");
    }

    auto &api = dpctl::detail::dpctl_capi::get();

    int v1_typenum = v1.get_typenum();
    int v2_typenum = v2.get_typenum();

    if ((v1_typenum != v2_typenum) ||
        ((v1_typenum != api.UAR_DOUBLE_) && (v1_typenum != api.UAR_FLOAT_) &&
         (v1_typenum != api.UAR_CDOUBLE_) && (v1_typenum != api.UAR_CFLOAT_)))
    {
        throw py::value_error(
            "Data types of vectors must be the same. "
            "Only real and complex floating types are supported.");
    }

    const char *v1_typeless_ptr = v1.get_data();
    const char *v2_typeless_ptr = v2.get_data();
    py::object res;

    if (v1_typenum == api.UAR_DOUBLE_) {
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
    else if (v1_typenum == api.UAR_FLOAT_) {
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
    else if (v1_typenum == api.UAR_CDOUBLE_) {
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
    else if (v1_typenum == api.UAR_CFLOAT_) {
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
        throw std::runtime_error("Type dispatch ran into trouble.");
    }

    return res;
}

int py_cg_solve(sycl::queue exec_q,
                dpctl::tensor::usm_ndarray Amat,
                dpctl::tensor::usm_ndarray bvec,
                dpctl::tensor::usm_ndarray xvec,
                double rs_tol,
                const std::vector<sycl::event> &depends = {})
{
    if (Amat.get_ndim() != 2 || bvec.get_ndim() != 1 || xvec.get_ndim() != 1) {
        throw py::value_error("Expecting a matrix and two vectors");
    }

    py::ssize_t n0 = Amat.get_shape(0);
    py::ssize_t n1 = Amat.get_shape(1);

    if (n0 != n1) {
        throw py::value_error("Matrix must be square.");
    }

    if (n0 != bvec.get_shape(0) || n0 != xvec.get_shape(0)) {
        throw py::value_error(
            "Dimensions of the matrix and vectors are not consistent.");
    }

    bool all_contig = (Amat.is_c_contiguous()) && (bvec.is_c_contiguous()) &&
                      (xvec.is_c_contiguous());
    if (!all_contig) {
        throw py::value_error("All inputs must be C-contiguous");
    }

    int A_typenum = Amat.get_typenum();
    int b_typenum = bvec.get_typenum();
    int x_typenum = xvec.get_typenum();

    if (A_typenum != b_typenum || A_typenum != x_typenum) {
        throw py::value_error("All arrays must have the same type");
    }

    if (!dpctl::utils::queues_are_compatible(
            exec_q, {Amat.get_queue(), bvec.get_queue(), xvec.get_queue()}))
    {
        throw std::runtime_error(
            "USM allocation queues are not the same as the execution queue");
    }

    const char *A_ch = Amat.get_data();
    const char *b_ch = bvec.get_data();
    char *x_ch = xvec.get_data();

    auto &api = dpctl::detail::dpctl_capi::get();

    if (A_typenum == api.UAR_DOUBLE_) {
        using T = double;
        int iters = cg_solver::cg_solve<T>(
            exec_q, n0, reinterpret_cast<const T *>(A_ch),
            reinterpret_cast<const T *>(b_ch), reinterpret_cast<T *>(x_ch),
            depends, static_cast<T>(rs_tol));

        return iters;
    }
    else if (A_typenum == api.UAR_FLOAT_) {
        using T = float;
        int iters = cg_solver::cg_solve<T>(
            exec_q, n0, reinterpret_cast<const T *>(A_ch),
            reinterpret_cast<const T *>(b_ch), reinterpret_cast<T *>(x_ch),
            depends, static_cast<T>(rs_tol));

        return iters;
    }
    else {
        throw std::runtime_error(
            "Unsupported data type. Use single or double precision.");
    }
}

PYBIND11_MODULE(_onemkl, m)
{
    m.def("gemv", &py_gemv, "Uses oneMKL to compute dot(matrix, vector)",
          py::arg("exec_queue"), py::arg("Amatrix"), py::arg("xvec"),
          py::arg("resvec"), py::arg("depends") = py::list());
    m.def("sub", &py_sub, "Subtraction: out = v1 - v2", py::arg("exec_queue"),
          py::arg("in1"), py::arg("in2"), py::arg("out"),
          py::arg("depends") = py::list());
    m.def("axpby_inplace", &py_axpby_inplace, "y = a * x + b * y",
          py::arg("exec_queue"), py::arg("a"), py::arg("x"), py::arg("b"),
          py::arg("y"), py::arg("depends") = py::list());
    m.def("norm_squared_blocking", &py_norm_squared_blocking, "norm(r)**2",
          py::arg("exec_queue"), py::arg("r"), py::arg("depends") = py::list());
    m.def("dot_blocking", &py_dot_blocking, "<v1, v2>", py::arg("exec_queue"),
          py::arg("v1"), py::arg("v2"), py::arg("depends") = py::list());

    m.def("cpp_cg_solve", &py_cg_solve,
          "Dispatch to call C++ implementation of cg_solve",
          py::arg("exec_queue"), py::arg("Amat"), py::arg("bvec"),
          py::arg("xvec"), py::arg("rs_squared_tolerance") = py::float_(1e-20),
          py::arg("depends") = py::list());
}
