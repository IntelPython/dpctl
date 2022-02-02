//===--- dpctl_sycl_kernel_interface.cpp - Implements C API for sycl::kernel =//
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
/// This file implements the functions declared in
/// dpctl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_kernel_interface.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_error_handlers.h"
#include "dpctl_string_utils.hpp"
#include <CL/sycl.hpp> /* Sycl headers */

using namespace cl::sycl;

namespace
{

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPCTLSyclKernelRef)

} /* end of anonymous namespace */

__dpctl_give const char *
DPCTLKernel_GetFunctionName(__dpctl_keep const DPCTLSyclKernelRef Kernel)
{
    if (!Kernel) {
        error_handler("Cannot get the number of arguments "
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return nullptr;
    }

    auto SyclKernel = unwrap(Kernel);
    auto kernel_name = SyclKernel->get_info<info::kernel::function_name>();
    if (kernel_name.empty())
        return nullptr;
    return dpctl::helper::cstring_from_string(kernel_name);
}

size_t DPCTLKernel_GetNumArgs(__dpctl_keep const DPCTLSyclKernelRef Kernel)
{
    if (!Kernel) {
        error_handler("Cannot get the number of arguments from "
                      "DPCTLSyclKernelRef as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return -1;
    }

    auto SyclKernel = unwrap(Kernel);
    auto num_args = SyclKernel->get_info<info::kernel::num_args>();
    return (size_t)num_args;
}

void DPCTLKernel_Delete(__dpctl_take DPCTLSyclKernelRef Kernel)
{
    delete unwrap(Kernel);
}
