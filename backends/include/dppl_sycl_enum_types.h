//===--- dppl_sycl_enum_types.h - DPPL-SYCL interface ---*---C++ -----*----===//
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
/// This header defines DPPL specficif enum types that wrap corresponding Sycl
/// enum classes. These enums are defined primarily so that Python extensions
/// that use DPPL do not have to include Sycl headers directly.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/ExternC.h"

DPPL_C_EXTERN_C_BEGIN

/*!
 * @brief Redefinition of DPC++-specific Sycl backend types.
 *
 */
enum DPPLSyclBackendType {
  DPPL_UNKNOWN_BACKEND = 0x0,
  DPPL_OPENCL = 1 << 16,
  DPPL_HOST = 1 << 15,
  DPPL_LEVEL_ZERO = 1 << 14,
  DPPL_CUDA = 1 << 13
};

/*!
 * @brief DPPL device types that are equivalent to Sycl's device_type.
 *
 */
enum DPPLSyclDeviceType {
  DPPL_CPU = 1 << 0,
  DPPL_GPU = 1 << 1,
  DPPL_ACCELERATOR = 1 << 2,
  DPPL_CUSTOM = 1 << 3,
  DPPL_AUTOMATIC = 1 << 4,
  DPPL_HOST_DEVICE = 1 << 5,
  DPPL_ALL = 1 << 6
  // IMP: before adding new values here look at DPPLSyclBackendType enum. The
  // values should not overlap.
};

/*!
 * @brief Supported types for kernel arguments to be passed to a Sycl kernel
 * using DPPL.
 *
 * \todo Add support for sycl::buffer
 *
 */
typedef enum {
  DPPL_CHAR,
  DPPL_SIGNED_CHAR,
  DPPL_UNSIGNED_CHAR,
  DPPL_SHORT,
  DPPL_INT,
  DPPL_UNSIGNED_INT,
  DPPL_UNSIGNED_INT8,
  DPPL_LONG,
  DPPL_UNSIGNED_LONG,
  DPPL_LONG_LONG,
  DPPL_UNSIGNED_LONG_LONG,
  DPPL_SIZE_T,
  DPPL_FLOAT,
  DPPL_DOUBLE,
  DPPL_LONG_DOUBLE,
  DPPL_VOID_PTR
} DPPLKernelArgType;

DPPL_C_EXTERN_C_END
