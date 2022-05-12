//==- _cg_solver.cpp - C++ impl of conjugate gradient linear solver  -==//
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

#pragma once

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

namespace cg_solver
{
namespace detail
{

template <typename T> class sub_kern;

template <typename T>
sycl::event sub(sycl::queue &q,
                size_t n,
                const T *v1,
                const T *v2,
                T *r,
                const std::vector<sycl::event> &depends = {})
{
    sycl::event r_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<sub_kern<T>>(sycl::range<1>{n}, [=](sycl::id<1> id) {
            auto i = id.get(0);
            r[i] = v1[i] - v2[i];
        });
    });

    return r_ev;
}

template <typename T> class axpy_inplace_kern;
template <typename T> class axpby_inplace_kern;

template <typename T>
sycl::event axpby_inplace(sycl::queue &q,
                          size_t nelems,
                          T a,
                          const T *x,
                          T b,
                          T *y,
                          const std::vector<sycl::event> depends = {})
{

    sycl::event res_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        if (b == T(1)) {
            cgh.parallel_for<axpy_inplace_kern<T>>(sycl::range<1>{nelems},
                                                   [=](sycl::id<1> id) {
                                                       auto i = id.get(0);
                                                       y[i] += a * x[i];
                                                   });
        }
        else {
            cgh.parallel_for<axpby_inplace_kern<T>>(
                sycl::range<1>{nelems}, [=](sycl::id<1> id) {
                    auto i = id.get(0);
                    y[i] = b * y[i] + a * x[i];
                });
        }
    });

    return res_ev;
}

template <typename T> class norm_squared_blocking_kern;

template <typename T>
T norm_squared_blocking(sycl::queue &q,
                        size_t nelems,
                        const T *r,
                        const std::vector<sycl::event> &depends = {})
{
    sycl::buffer<T, 1> sum_sq_buf(sycl::range<1>{1});

    q.submit([&](sycl::handler &cgh) {
         cgh.depends_on(depends);
         auto sum_sq_reduction = sycl::reduction(
             sum_sq_buf, cgh, sycl::plus<T>(),
             {sycl::property::reduction::initialize_to_identity{}});
         cgh.parallel_for<norm_squared_blocking_kern<T>>(
             sycl::range<1>{nelems}, sum_sq_reduction,
             [=](sycl::id<1> id, auto &sum_sq) {
                 auto i = id.get(0);
                 sum_sq += r[i] * r[i];
             });
     }).wait_and_throw();

    sycl::host_accessor ha(sum_sq_buf);
    return ha[0];
}

template <typename T> class complex_norm_squared_blocking_kern;

template <typename T>
T complex_norm_squared_blocking(sycl::queue &q,
                                size_t nelems,
                                const std::complex<T> *r,
                                const std::vector<sycl::event> &depends = {})
{
    sycl::buffer<T, 1> sum_sq_buf(sycl::range<1>{1});

    q.submit([&](sycl::handler &cgh) {
         cgh.depends_on(depends);
         auto sum_sq_reduction = sycl::reduction(
             sum_sq_buf, cgh, sycl::plus<T>(),
             {sycl::property::reduction::initialize_to_identity{}});
         cgh.parallel_for<complex_norm_squared_blocking_kern<T>>(
             sycl::range<1>{nelems}, sum_sq_reduction,
             [=](sycl::id<1> id, auto &sum_sq) {
                 auto i = id.get(0);
                 sum_sq +=
                     r[i].real() * r[i].real() + r[i].imag() * r[i].imag();
             });
     }).wait_and_throw();

    sycl::host_accessor ha(sum_sq_buf);
    return ha[0];
}

} // namespace detail

template <typename T>
int cg_solve(sycl::queue exec_q,
             std::int64_t n,
             const T *Amat,
             const T *bvec,
             T *sol_vec,
             const std::vector<sycl::event> &depends = {},
             T rs_threshold = T(1e-20))
{
    T *x_vec = sol_vec;
    sycl::event fill_ev = exec_q.fill<T>(x_vec, T(0), n, depends);

    // n for r, n for p, n for Ap and 1 for pAp_dot_dev
    T *r = sycl::malloc_device<T>(3 * n + 1, exec_q);
    T *p = r + n;
    T *Ap = p + n;
    T *pAp_dot_dev = Ap + n;

    sycl::event copy_to_r_ev = exec_q.copy<T>(bvec, r, n, depends);
    sycl::event copy_to_p_ev = exec_q.copy<T>(bvec, p, n, depends);
    T rs_old = detail::norm_squared_blocking(exec_q, n, r, {copy_to_r_ev});

    std::int64_t max_iters = n;

    if (rs_old < rs_threshold) {
        sycl::free(r, exec_q);
        return 0;
    }

    int converged_at = max_iters;
    sycl::event e_p = copy_to_p_ev;
    sycl::event e_x = fill_ev;

    for (std::int64_t i = 0; i < max_iters; ++i) {
        sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
            exec_q, oneapi::mkl::transpose::N, n, n, T(1), Amat, n, p, 1, T(0),
            Ap, 1, {e_p});

        sycl::event pAp_dot_ev = oneapi::mkl::blas::row_major::dot(
            exec_q, n, p, 1, Ap, 1, pAp_dot_dev, {e_p, gemv_ev});

        T pAp_dot_host{};
        exec_q.copy<T>(pAp_dot_dev, &pAp_dot_host, 1, {pAp_dot_ev})
            .wait_and_throw();
        T alpha = rs_old / pAp_dot_host;

        // x = x + alpha * p
        sycl::event x_update_ev =
            detail::axpby_inplace(exec_q, n, alpha, p, T(1), x_vec, {e_x});

        // r = r - alpha * Ap
        sycl::event r_update_ev =
            detail::axpby_inplace(exec_q, n, -alpha, Ap, T(1), r);

        T rs_new = detail::norm_squared_blocking(exec_q, n, r, {r_update_ev});

        if (rs_new < rs_threshold) {
            converged_at = i;
            x_update_ev.wait();
            break;
        }

        T beta = rs_new / rs_old;

        // p = r + beta * p
        e_p = detail::axpby_inplace(exec_q, n, T(1), r, beta, p, {r_update_ev});
        e_x = x_update_ev;

        rs_old = rs_new;
    }

    sycl::free(r, exec_q);
    return converged_at;
}

} // namespace cg_solver
