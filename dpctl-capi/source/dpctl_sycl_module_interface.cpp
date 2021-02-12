//===- dpctl_sycl_module_interface.cpp - Implements C API for sycl::module --=//
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
/// Implementation of functions declared in dpctl_sycl_module_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_module_interface.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp> /* Sycl headers       */

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(program, DPCTLSyclProgramRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPCTLSyclKernelRef)

} /* end of anonymous namespace */

__dpctl_give DPCTLSyclKernelRef
DPCTLProgram_GetKernel(__dpctl_keep DPCTLSyclProgramRef PRef,
                       __dpctl_keep const char *KernelName)
{
    if (!PRef) {
        // \todo record error
        return nullptr;
    }
    auto SyclProgram = unwrap(PRef);
    if (!KernelName) {
        // \todo record error
        return nullptr;
    }
    std::string name = KernelName;
    try {
        auto SyclKernel = new kernel(SyclProgram->get_kernel(name));
        return wrap(SyclKernel);
    } catch (invalid_object_error &e) {
        // \todo record error
        std::cerr << e.what() << '\n';
        return nullptr;
    }
}

bool DPCTLProgram_HasKernel(__dpctl_keep DPCTLSyclProgramRef PRef,
                            __dpctl_keep const char *KernelName)
{
    if (!PRef) {
        // \todo handle error
        return false;
    }

    auto SyclProgram = unwrap(PRef);
    try {
        return SyclProgram->has_kernel(KernelName);
    } catch (invalid_object_error &e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

void DPCTLProgram_Delete(__dpctl_take DPCTLSyclProgramRef PRef)
{
    delete unwrap(PRef);
}
