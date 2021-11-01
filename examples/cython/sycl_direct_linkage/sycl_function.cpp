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
/// using `sycl::buffer`. The routine solves the task by calling BLAS function
//  GEMV from Intel(R) Math Kernel Library.
///
//===----------------------------------------------------------------------===//

#include "sycl_function.hpp"
#include "mkl.h"
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

int c_columnwise_total(cl::sycl::queue &q,
                       size_t n,
                       size_t m,
                       double *mat,
                       double *ct)
{
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
                      << e.what() << std::endl
                      << "OpenCL status: " << e.get_cl_code() << std::endl;
            goto cleanup;
        }

        try {
            oneapi::mkl::blas::row_major::gemv(
                q, oneapi::mkl::transpose::trans, n, m, double(1.0), mat_buffer,
                m, ones_buffer, 1, double(0.0), ct_buffer, 1);
            q.wait();
        } catch (sycl::exception const &e) {
            std::cout << "\t\tCaught synchronous SYCL exception during GEMV:\n"
                      << e.what() << std::endl
                      << "OpenCL status: " << e.get_cl_code() << std::endl;
            goto cleanup;
        }
    }

    free(ones);
    return 0;

cleanup:
    free(ones);
    return -1;
}
