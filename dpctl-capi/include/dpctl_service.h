//===- dpctl_service.h - C API for service functions   -*-C++-*- ===//
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
/// This header defines dpctl service functions.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"

DPCTL_C_EXTERN_C_BEGIN
/**
 * @defgroup Service Service functions
 */

/*!
 * @brief Get version of DPC++ toolchain the library was compiled with.
 *
 * @return A C string containing the version of DPC++ toolchain.
 * @ingroup Service
 */
DPCTL_API
__dpctl_give const char *DPCTLService_GetDPCPPVersion(void);

DPCTL_C_EXTERN_C_END
