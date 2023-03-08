//=- use_sycl_buffer.cpp - Example of SYCL code to be called from Cython  =//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2023 Intel Corporation
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

#pragma once

#include <CL/sycl.hpp>
#include <iostream>

inline size_t upper_multiple(size_t n, size_t wg)
{
    return wg * ((n + wg - 1) / wg);
}

template <typename dataT>
void columnwise_total(sycl::queue q,
                      size_t n,
                      size_t m,
                      const dataT *mat,
                      dataT *ct)
{
    sycl::buffer<dataT, 2> mat_buffer = sycl::buffer(mat, sycl::range<2>(n, m));
    sycl::buffer<dataT, 1> ct_buffer = sycl::buffer(ct, sycl::range<1>(m));

    q.submit([&](sycl::handler &h) {
        sycl::accessor ct_acc{
            ct_buffer, h, sycl::write_only, {sycl::property::no_init{}}};
        h.parallel_for(sycl::range<1>(m),
                       [=](sycl::id<1> i) { ct_acc[i] = dataT(0); });
    });

    constexpr size_t wg = 256;

    q.submit([&](sycl::handler &h) {
        sycl::accessor mat_acc{mat_buffer, h, sycl::read_only};
        sycl::accessor ct_acc{ct_buffer, h};

        sycl::range<2> global{upper_multiple(n, wg), m};
        sycl::range<2> local{wg, 1};

        h.parallel_for(
            sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
                size_t i = it.get_global_id(0);
                size_t j = it.get_global_id(1);
                dataT group_sum = sycl::reduce_over_group(
                    it.get_group(),
                    (i < n) ? mat_acc[it.get_global_id()] : dataT(0),
                    std::plus<dataT>());
                if (it.get_local_id(0) == 0) {
                    sycl::atomic_ref<dataT, sycl::memory_order::relaxed,
                                     sycl::memory_scope::system,
                                     sycl::access::address_space::global_space>(
                        ct_acc[j]) += group_sum;
                }
            });
    });

    return;
}
