//=- use_sycl_buffer.cpp - Example of SYCL code to be called from Cython  =//
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
/// This file implements SYCL code to compute columnwise total of a matrix,
/// provided as host C-contiguous allocation. SYCL kernels access this memory
/// using `sycl::buffer`. Two routines are provided. One solves the task by
/// calling BLAS function GEMV from Intel(R) Math Kernel Library, the other
/// performs the computation using DPC++ reduction group function and atomics.
///
//===----------------------------------------------------------------------===//

#include "use_sycl_buffer.h"
#include "dpctl_sycl_interface.h"
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

int c_columnwise_total(DPCTLSyclQueueRef q_ref,
                       size_t n,
                       size_t m,
                       double *mat,
                       double *ct)
{

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    sycl::buffer<double, 1> mat_buffer =
        sycl::buffer(mat, sycl::range<1>(n * m));
    sycl::buffer<double, 1> ct_buffer = sycl::buffer(ct, sycl::range<1>(m));

    double *ones = reinterpret_cast<double *>(malloc(n * sizeof(double)));
    {
        sycl::buffer<double, 1> ones_buffer =
            sycl::buffer(ones, sycl::range<1>(n));

        try {
            auto ev = q.submit([&](sycl::handler &cgh) {
                auto ones_acc =
                    ones_buffer.get_access<sycl::access::mode::read_write>(cgh);
                cgh.fill(ones_acc, double(1.0));
            });

            ev.wait_and_throw();
        } catch (sycl::exception const &e) {
            std::cout << "\t\tCaught synchronous SYCL exception during fill:\n"
                      << e.what() << std::endl;
            goto cleanup;
        }

        try {
            oneapi::mkl::blas::row_major::gemv(
                q, oneapi::mkl::transpose::trans, n, m, double(1.0), mat_buffer,
                m, ones_buffer, 1, double(0.0), ct_buffer, 1);
            q.wait();
        } catch (sycl::exception const &e) {
            std::cout << "\t\tCaught synchronous SYCL exception during GEMV:\n"
                      << e.what() << std::endl;
            goto cleanup;
        }
    }

    free(ones);
    return 0;

cleanup:
    free(ones);
    return -1;
}

inline size_t upper_multiple(size_t n, size_t wg)
{
    return wg * ((n + wg - 1) / wg);
}

int c_columnwise_total_no_mkl(DPCTLSyclQueueRef q_ref,
                              size_t n,
                              size_t m,
                              double *mat,
                              double *ct)
{

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    sycl::buffer<double, 2> mat_buffer =
        sycl::buffer(mat, sycl::range<2>(n, m));
    sycl::buffer<double, 1> ct_buffer = sycl::buffer(ct, sycl::range<1>(m));

    auto e = q.submit([&](sycl::handler &h) {
        sycl::accessor ct_acc{ct_buffer, h, sycl::write_only};
        h.parallel_for(sycl::range<1>(m),
                       [=](sycl::id<1> i) { ct_acc[i] = 0.0; });
    });

    constexpr size_t wg = 256;
    auto e2 = q.submit([&](sycl::handler &h) {
        sycl::accessor mat_acc{mat_buffer, h, sycl::read_only};
        sycl::accessor ct_acc{ct_buffer, h};
        h.depends_on(e);

        sycl::range<2> global{upper_multiple(n, wg), m};
        sycl::range<2> local{wg, 1};

        h.parallel_for(
            sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
                size_t i = it.get_global_id(0);
                size_t j = it.get_global_id(1);
                double group_sum = sycl::reduce_over_group(
                    it.get_group(), (i < n) ? mat_acc[it.get_global_id()] : 0.0,
                    std::plus<double>());
                if (it.get_local_id(0) == 0) {
                    sycl::ext::oneapi::atomic_ref<
                        double, sycl::memory_order::relaxed,
                        sycl::memory_scope::system,
                        sycl::access::address_space::global_space>(ct_acc[j]) +=
                        group_sum;
                }
            });
    });

    e2.wait_and_throw();
    return 0;
}
