//=== dpctl_sycl_device_selector_interface.h - device_selector C API -*-C++-*-//
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
/// This header declares C contructors for the various SYCL device_selector
/// classes.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup DeviceSelectors C API for sycl::device_selector classess.
 */

/*!
 * @brief Returns an opaque wrapper for sycl::accelerator_selector object.
 *
 * @return An opaque pointer to a sycl::accelerator_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLAcceleratorSelector_Create();

/*!
 * @brief Returns an opaque wrapper for sycl::default_selector object.
 *
 * @return An opaque pointer to a sycl::default_selector object.
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLDefaultSelector_Create();

/*!
 * @brief Returns an opaque wrapper for sycl::cpu_selector object.
 *
 * @return An opaque pointer to a sycl::cpu_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLCPUSelector_Create();

/*!
 * @brief Returns an opaque wrapper for sycl::ONEAPI::filter_selector object
 * based on the passed in filter string.
 *
 * @param    filter_str     A C string providing a filter based on which to
 *                          create a device_selector.
 * @return   An opaque pointer to a sycl::ONEAPI::filter_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef
DPCTLFilterSelector_Create(__dpctl_keep const char *filter_str);

/*!
 * @brief Returns an opaque wrapper for sycl::gpu_selector object.
 *
 * @return An opaque pointer to a sycl::gpu_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLGPUSelector_Create();

/*!
 * @brief Returns an opaque wrapper for sycl::host_selector object.
 *
 * @return An opaque pointer to a sycl::host_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLHostSelector_Create();

/*!
 * @brief Deletes the DPCTLSyclDeviceSelectorRef after casting it to a
 * sycl::device_selector.
 *
 * @param    DSRef An opaque DPCTLSyclDeviceSelectorRef pointer that would be
 *                 freed.
 * @ingroup DeviceSelectors
 */
DPCTL_API
void DPCTLDeviceSelector_Delete(__dpctl_take DPCTLSyclDeviceSelectorRef DSRef);

DPCTL_C_EXTERN_C_END
