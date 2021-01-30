//===--- dpctl_sycl_kernel_interface.cpp - Implements C API for sycl::kernel =//
//
//                      Data Parallel Control (dpCtl)
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
/// This file implements the functions declared in
/// dpctl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_kernel_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp> /* Sycl headers */

using namespace cl::sycl;

namespace
{

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPCTLSyclKernelRef)

} /* end of anonymous namespace */

__dpctl_give const char*
DPCTLKernel_GetFunctionName (__dpctl_keep const DPCTLSyclKernelRef Kernel)
{
    if(!Kernel) {
        // \todo record error
        return nullptr;
    }

    auto SyclKernel = unwrap(Kernel);
    auto kernel_name = SyclKernel->get_info<info::kernel::function_name>();
    if(kernel_name.empty())
        return nullptr;
    auto cstr_len = kernel_name.length()+1;
    auto cstr_name = new char[cstr_len];
#ifdef _WIN32
    strncpy_s(cstr_name, cstr_len, kernel_name.c_str(), cstr_len);
#else
    std::strncpy (cstr_name, kernel_name.c_str(), cstr_len);
#endif
    return cstr_name;
}

size_t
DPCTLKernel_GetNumArgs (__dpctl_keep const DPCTLSyclKernelRef Kernel)
{
    if(!Kernel) {
        // \todo record error
        return -1;
    }

    auto SyclKernel = unwrap(Kernel);
    auto num_args = SyclKernel->get_info<info::kernel::num_args>();
    return (size_t)num_args;
}

void
DPCTLKernel_Delete (__dpctl_take DPCTLSyclKernelRef Kernel)
{
    delete unwrap(Kernel);
}
