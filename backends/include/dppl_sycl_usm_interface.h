//===--- dppl_sycl_usm_interface.h - DPPL-SYCL interface ---*---C++ -*---===//
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
/// This header declares a C interface to sycl::usm interface functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dppl_data_types.h"
#include "dppl_sycl_types.h"
#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Crete USM shared memory.
 *
 * @return The pointer to USM shared memory.
 */
DPPL_API
__dppl_give DPPLMemoryUSMSharedRef
DPPLmalloc_shared (size_t size, __dppl_keep const DPPLSyclQueueRef QRef);

/*!
 * @brief Free USM memory.
 *
 */
DPPL_API
void DPPLfree (__dppl_take DPPLMemoryUSMSharedRef MRef, __dppl_keep const DPPLSyclQueueRef QRef);

DPPL_C_EXTERN_C_END
