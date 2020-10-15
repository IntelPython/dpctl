//===------ dppl_sycl_kernel_interface.cpp - dpctl-C_API  ---*--- C++ --*--===//
//
//               Data Parallel Control Library (dpCtl)
//
// Copyright 2020 Intel Corporation
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
/// dppl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_kernel_interface.h"
#include "Support/CBindingWrapping.h"

#include <CL/sycl.hpp> /* Sycl headers */

using namespace cl::sycl;

namespace {

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(kernel, DPPLSyclKernelRef)

} /* end of anonymous namespace */

__dppl_give const char *
DPPLKernel_GetFunctionName(__dppl_keep const DPPLSyclKernelRef Kernel) {
  if (!Kernel) {
    // \todo record error
    return nullptr;
  }

  auto SyclKernel = unwrap(Kernel);
  auto kernel_name = SyclKernel->get_info<info::kernel::function_name>();
  if (kernel_name.empty())
    return nullptr;
  auto cstr_name = new char[kernel_name.length() + 1];
  std::strcpy(cstr_name, kernel_name.c_str());
  return cstr_name;
}

size_t DPPLKernel_GetNumArgs(__dppl_keep const DPPLSyclKernelRef Kernel) {
  if (!Kernel) {
    // \todo record error
    return -1;
  }

  auto SyclKernel = unwrap(Kernel);
  auto num_args = SyclKernel->get_info<info::kernel::num_args>();
  return (size_t)num_args;
}

void DPPLKernel_Delete(__dppl_take DPPLSyclKernelRef Kernel) {
  delete unwrap(Kernel);
}
