//===--- dppl_sycl_context_interface.cpp - DPPL-SYCL interface --*- C++ -*-===//
//
//               Python Data Parallel Processing Library (PyDPPL)
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
/// This file implements the data types and functions declared in
/// dppl_sycl_context_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_context_interface.h"
#include "Support/CBindingWrapping.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

namespace
{
 // Create wrappers for C Binding types (see CBindingWrapping.h).
 DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPPLSyclContextRef)
} /* end of anonymous namespace */

/*!
 * @brief
 *
 * @param    CtxtRef        My Param doc
 * @return   {return}       My Param doc
 */
bool DPPLContext_IsHost (__dppl_keep const DPPLSyclContextRef CtxRef)
{
    return unwrap(CtxRef)->is_host();
}

/*!
 * @brief
 *
 * @param    CtxtRef        My Param doc
 */
void DPPLContext_Delete (__dppl_take DPPLSyclContextRef CtxRef)
{
    delete unwrap(CtxRef);
}
