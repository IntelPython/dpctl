//=- sycl_blackscholes.cpp - Example of SYCL code to be called from Cython  =//
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
/// This file implements SYCL code to price European vanilla options using
/// Black-Scholes formula, as well as code to generate option parameters using
/// SYCL device random number generation library from Intel(R) Math Kernel
/// Library.
///
//===----------------------------------------------------------------------===//

#include "sycl_blackscholes.hpp"
#include "dpctl_sycl_interface.h"
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng/device.hpp>

template <typename T> class black_scholes_kernel;

constexpr int n_params = 5;
constexpr int n_params_next_pow2 = 8;

constexpr int n_prices = 2;
constexpr int PRICE = 0;
constexpr int STRIKE = 1;
constexpr int MATURITY = 2;
constexpr int RATE = 3;
constexpr int VOLATILITY = 4;
constexpr int CALL = 0;
constexpr int PUT = 1;

template <typename T>
extern void
cpp_blackscholes(DPCTLSyclQueueRef q_ptr, size_t n_opts, T *params, T *callput)
{
    using data_t = T;

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ptr));

    auto ctx = q.get_context();
    {
        sycl::usm::alloc params_type = sycl::get_pointer_type(params, ctx);
        if (params_type != sycl::usm::alloc::shared) {
            throw std::runtime_error("Input option_params to cpp_blackscholes "
                                     "is not a USM-shared pointer.");
        }
    }
    {
        sycl::usm::alloc callput_type = sycl::get_pointer_type(callput, ctx);
        if (callput_type != sycl::usm::alloc::shared) {
            throw std::runtime_error("Input callput to cpp_blackscholes is not "
                                     "a USM-shared pointer.");
        }
    }

    auto e = q.submit([&](sycl::handler &cgh) {
        data_t zero = data_t(0), one = data_t(1), two = data_t(2);
        data_t quarter = one / data_t(4);
        data_t half = one / two;

            cgh.parallel_for<class black_scholes_kernel<T>>(
		sycl::range<1>(n_opts),
		[=](sycl::id<1> idx)
            {
                const size_t i = n_params * idx[0];
                const data_t opt_price = params[i + PRICE];
                const data_t opt_strike = params[i + STRIKE];
                const data_t opt_maturity = params[i + MATURITY];
                const data_t opt_rate = params[i + RATE];
                const data_t opt_volatility = params[i + VOLATILITY];
                data_t a, b, c, y, z, e, d1, d1c, d2, d2c, w1, w2;
                data_t mr = -opt_rate,
                       sig_sig_two = two * opt_volatility * opt_volatility;

                a = cl::sycl::log(opt_price / opt_strike);
                b = opt_maturity * mr;
                z = opt_maturity * sig_sig_two;

                c = quarter * z;
                e = cl::sycl::exp(b);
                y = cl::sycl::rsqrt(z);

                a = b - a;
                w1 = (a - c) * y;
                w2 = (a + c) * y;

                if (w1 < zero) {
                    d1 = cl::sycl::erfc(w1) * half;
                    d1c = one - d1;
                }
                else {
                    d1c = cl::sycl::erfc(-w1) * half;
                    d1 = one - d1c;
                }
                if (w2 < zero) {
                    d2 = cl::sycl::erfc(w2) * half;
                    d2c = one - d2;
                }
                else {
                    d2c = cl::sycl::erfc(-w2) * half;
                    d2 = one - d2c;
                }

                e *= opt_strike;
                data_t call_price = opt_price * d1 - e * d2;
                data_t put_price = e * d2c - opt_price * d1c;

                const size_t callput_i = n_prices * idx[0];
                callput[callput_i + CALL] = call_price;
                callput[callput_i + PUT] = put_price;
                });
    });

    e.wait_and_throw();

    return;
}

template <typename T>
void cpp_populate_params(DPCTLSyclQueueRef q_ptr,
                         size_t n_opts,
                         T *params,
                         T pl,
                         T ph,
                         T sl,
                         T sh,
                         T tl,
                         T th,
                         T rl,
                         T rh,
                         T vl,
                         T vh,
                         int seed)
{
    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ptr));

    auto ctx = q.get_context();
    {
        sycl::usm::alloc params_type = sycl::get_pointer_type(params, ctx);
        if (params_type != sycl::usm::alloc::shared) {
            throw std::runtime_error("Input option_params to cpp_blackscholes "
                                     "is not a USM-shared pointer.");
        }
    }

    sycl::event e = q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>(n_opts), [=](sycl::item<1> idx) {
            size_t i = n_params * idx.get_id(0);
            size_t j = n_params_next_pow2 * idx.get_id(0);

            // create engine to sample 5 parameters per workers
            oneapi::mkl::rng::device::philox4x32x10<n_params_next_pow2> engine(
                seed, j);
            oneapi::mkl::rng::device::uniform<T> distr;

            sycl::vec<T, n_params_next_pow2> res =
                oneapi::mkl::rng::device::generate(distr, engine);

            {
                const int pos = PRICE;
                auto u = res[pos];
                params[i + pos] = pl * u + ph * (T(1) - u);
            }
            {
                const int pos = STRIKE;
                auto u = res[pos];
                params[i + pos] = sl * u + sh * (T(1) - u);
            }
            {
                const int pos = MATURITY;
                auto u = res[pos];
                params[i + pos] = tl * u + th * (T(1) - u);
            }
            {
                const int pos = RATE;
                auto u = res[pos];
                params[i + pos] = rl * u + rh * (T(1) - u);
            }
            {
                const int pos = VOLATILITY;
                auto u = res[pos];
                params[i + pos] = vl * u + vh * (T(1) - u);
            }
        });
    });

    e.wait_and_throw();
}

// instantation for object files to not be empty

template void cpp_blackscholes<double>(DPCTLSyclQueueRef q_ptr,
                                       size_t n_opts,
                                       double *params,
                                       double *callput);
template void cpp_blackscholes<float>(DPCTLSyclQueueRef q_ptr,
                                      size_t n_opts,
                                      float *params,
                                      float *callput);

template void cpp_populate_params<double>(DPCTLSyclQueueRef q_ptr,
                                          size_t n_opts,
                                          double *params,
                                          double pl,
                                          double ph,
                                          double sl,
                                          double sh,
                                          double tl,
                                          double th,
                                          double rl,
                                          double rh,
                                          double vl,
                                          double vh,
                                          int seed);
template void cpp_populate_params<float>(DPCTLSyclQueueRef q_ptr,
                                         size_t n_opts,
                                         float *params,
                                         float pl,
                                         float ph,
                                         float sl,
                                         float sh,
                                         float tl,
                                         float th,
                                         float rl,
                                         float rh,
                                         float vl,
                                         float vh,
                                         int seed);
