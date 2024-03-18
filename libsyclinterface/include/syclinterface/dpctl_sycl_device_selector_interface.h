//=== dpctl_sycl_device_selector_interface.h - device_selector C API -*-C++-*-//
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
/// This header declares C constructors for the various SYCL device_selector
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
 * @defgroup DeviceSelectors Device selection
 */

/*!
 * @brief Returns an opaque wrapper for sycl::accelerator_selector object.
 *
 * @return An opaque pointer to a sycl::accelerator_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLAcceleratorSelector_Create(void);

/*!
 * @brief Returns an opaque wrapper for sycl::default_selector object.
 *
 * @return An opaque pointer to a sycl::default_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLDefaultSelector_Create(void);

/*!
 * @brief Returns an opaque wrapper for sycl::cpu_selector object.
 *
 * @return An opaque pointer to a sycl::cpu_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLCPUSelector_Create(void);

/*!
 * @brief Returns an opaque wrapper for sycl::ext::oneapi::filter_selector
 * object based on the passed in filter string.
 *
 * @param    filter_str     A C string providing a filter based on which to
 *                          create a device selector.
 * @return   An opaque pointer to a sycl::ext::oneapi::filter_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef
DPCTLFilterSelector_Create(__dpctl_keep const char *filter_str);

/*!
 * @brief Returns an opaque wrapper for dpctl_gpu_selector object.
 *
 * @return An opaque pointer to a dpctl_gpu_selector object.
 * @ingroup DeviceSelectors
 */
DPCTL_API
__dpctl_give DPCTLSyclDeviceSelectorRef DPCTLGPUSelector_Create(void);

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

/*!
 *@brief Scores the device specified by DRef by device selector specified by
 *DSRef.
 *
 * @param    DSRef An opaque DPCTLSyclDeviceSelectorRef pointer.
 * @param    DRef An opaque DPCTLSyclDeviceRef pointer.
 *
 * @return A integer score. The negative value indicates select rejected the
 *device.
 * @ingroup DeviceSelectors
 */
DPCTL_API
int DPCTLDeviceSelector_Score(__dpctl_keep DPCTLSyclDeviceSelectorRef DSRef,
                              __dpctl_keep DPCTLSyclDeviceRef DRef);

DPCTL_C_EXTERN_C_END
