//===--- dppl_sycl_event_interface.h - DPPL-SYCL interface ---*---C++ -*---===//
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
/// This header declares a C API to a sub-set of the sycl::event interface.
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
 * @brief C-API wrapper for sycl::event.wait.
 *
 * @param    ERef           An opaque DPPLSyclEventRef pointer on which to wait.
 */
DPPL_API
void DPPLEvent_Wait (__dppl_keep DPPLSyclEventRef ERef);

/*!
 * @brief Deletes the DPPLSyclEventRef after casting it to a sycl::event.
 *
 * @param    ERef           An opaque DPPLSyclEventRef pointer that would be
 *                          freed.
 */
DPPL_API
void
DPPLEvent_Delete (__dppl_take DPPLSyclEventRef ERef);

DPPL_C_EXTERN_C_END
