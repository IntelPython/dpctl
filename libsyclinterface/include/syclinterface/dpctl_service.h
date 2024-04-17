//===- dpctl_service.h - C API for service functions   -*-C++-*- ===//
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

/*!
 * @brief Initialize logger if compiled to use logger, no-op otherwise.
 *
 * @param app_name  C-string for application name reflected in the log.
 * @param log_dir   C-string for directory where log files are placed.
 * @ingroup Service
 */
DPCTL_API
void DPCTLService_InitLogger(const char *app_name, const char *log_dir);

/*!
 * @brief Finalize logger if enabled, no-op otherwise.
 *
 * @ingroup Service
 */
DPCTL_API
void DPCTLService_ShutdownLogger(void);

DPCTL_C_EXTERN_C_END
