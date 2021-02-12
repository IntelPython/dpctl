//===--- dpctl_sycl_module_manager.h - Helpers for sycl::module  -*-C++-*- ===//
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
/// Helper functions to create sycl::module (currently sycl::program) objects.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Create a Sycl program from an OpenCL SPIR-V binary file.
 *
 * Sycl 1.2 does not expose any method to create a sycl::program from a SPIR-V
 * IL file. To get around this limitation, we first creare a SYCL
 * interoperability program and then create a SYCL program from the
 * interoperability program. Currently, interoperability programs can be created
 * for OpenCL and Level-0 backends.
 *
 * The feature to create a Sycl kernel from a SPIR-V IL binary will be available
 * in Sycl 2.0 at which point this function may become deprecated.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    IL             SPIR-V binary
 * @param    Length         The size of the IL binary in bytes.
 * @param    CompileOpts    Optional compiler flags used when compiling the
 *                          SPIR-V binary.
 * @return   A new SyclProgramRef pointer if the program creation succeeded,
 *           else returns NULL.
 */
DPCTL_API
__dpctl_give DPCTLSyclProgramRef
DPCTLProgram_CreateFromSpirv(__dpctl_keep const DPCTLSyclContextRef Ctx,
                             __dpctl_keep const void *IL,
                             size_t Length,
                             const char *CompileOpts);

/*!
 * @brief Create a Sycl program from an OpenCL kernel source string.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    Source         OpenCL source string
 * @param    CompileOpts    Extra compiler flags (refer Sycl spec.)
 * @return   A new SyclProgramRef pointer if the program creation succeeded,
 *           else returns NULL.
 */
DPCTL_API
__dpctl_give DPCTLSyclProgramRef
DPCTLProgram_CreateFromOCLSource(__dpctl_keep const DPCTLSyclContextRef Ctx,
                                 __dpctl_keep const char *Source,
                                 __dpctl_keep const char *CompileOpts);

DPCTL_C_EXTERN_C_END
