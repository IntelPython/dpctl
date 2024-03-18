//===----------- dpctl_utils.h - Helper functions                -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
/// This file defines common helper functions used in other places in dpctl.
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Deletes the C String argument.
 *
 * @param    str            C string to be deleted
 */
DPCTL_API
void DPCTLCString_Delete(__dpctl_take const char *str);

/*!
 * @brief Deletes an array of size_t elements.
 *
 * @param    arr            Array to be deleted.
 */
DPCTL_API
void DPCTLSize_t_Array_Delete(__dpctl_take size_t *arr);

DPCTL_C_EXTERN_C_END
