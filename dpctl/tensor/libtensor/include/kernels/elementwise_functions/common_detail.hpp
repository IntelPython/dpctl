//=== common_detail.hpp -                                     - *-C++-*--/===//
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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines common code for elementwise tensor operations.
//===---------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace elementwise_detail
{

template <typename T> class populate_padded_vec_krn;

template <typename T>
sycl::event
populate_padded_vector(sycl::queue &exec_q,
                       const T *vec,
                       std::size_t vec_sz,
                       T *padded_vec,
                       size_t padded_vec_sz,
                       const std::vector<sycl::event> &dependent_events)
{
    sycl::event populate_padded_vec_ev = exec_q.submit([&](sycl::handler &cgh) {
        // ensure vec contains actual data
        cgh.depends_on(dependent_events);

        sycl::range<1> gRange{padded_vec_sz};

        cgh.parallel_for<class populate_padded_vec_krn<T>>(
            gRange, [=](sycl::id<1> id) {
                std::size_t i = id[0];
                padded_vec[i] = vec[i % vec_sz];
            });
    });

    return populate_padded_vec_ev;
}

} // end of namespace elementwise_detail
} // end of namespace kernels
} // end of namespace tensor
} // end of namespace dpctl
