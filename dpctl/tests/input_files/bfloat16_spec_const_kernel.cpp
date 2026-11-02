//===-- bfloat16_spec_const_kernel.cpp - source of a test SPIR-V module ---===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2026 Intel Corporation
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
/// Source of bfloat16_spec_const_kernel.spv, which is used by
/// dpctl/tests/test_sycl_compiler.py to test that a specialization constant of
/// the type of the DPC++ ``bfloat16`` extension can be set from dpctl.
///
/// The SPIR-V module was produced with
///
///   icpx -fsycl -O2 -save-offload-code=. bfloat16_spec_const_kernel.cpp
///
/// which writes the device image of the kernel next to the executable.
///
/// The value of the specialization constant is widened to a float by hand,
/// rather than by converting the bfloat16, so that the module does not import
/// the DPC++ bfloat16 devicelib: the SPIR-V interoperability path does not link
/// device libraries, and a module with an unresolved import can not be run.
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

using bfloat16_t = sycl::ext::oneapi::bfloat16;

// the default value has to be the value-initialized bfloat16, as bfloat16 has
// no constexpr constructor
constexpr sycl::specialization_id<bfloat16_t> bf16_spec_const{};

class Bfloat16SpecConstKernel;

int main()
{
    sycl::queue q;
    const std::size_t n = 4;
    float *x = sycl::malloc_shared<float>(n, q);
    float *y = sycl::malloc_shared<float>(n, q);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 0.0f;
    }

    q.submit([&](sycl::handler &cgh) {
         cgh.set_specialization_constant<bf16_spec_const>(bfloat16_t(1.5f));
         cgh.parallel_for<Bfloat16SpecConstKernel>(
             sycl::range<1>(n), [=](sycl::id<1> i, sycl::kernel_handler kh) {
                 const float value = x[i];
                 const bfloat16_t factor =
                     kh.get_specialization_constant<bf16_spec_const>();
                 const std::uint32_t bits =
                     static_cast<std::uint32_t>(
                         sycl::bit_cast<std::uint16_t>(factor))
                     << 16;
                 y[i] = value * sycl::bit_cast<float>(bits);
             });
     }).wait();

    sycl::free(x, q);
    sycl::free(y, q);
    return 0;
}
