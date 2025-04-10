//===- dpctl_sycl_kernel_bundle_interface.h - C API for
//     sycl::kernel_bundle<sycl::bundle_state::executable>      -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This header declares a C API to create Sycl interoperability programs for
/// OpenCL and Level Zero driver API.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

/**
 * @defgroup KernelBundleInterface Kernel_bundle class C wrapper
 */

/*!
 * @brief Create a Sycl kernel_bundle from an OpenCL SPIR-V binary file.
 *
 * Uses SYCL2020 interoperability layer to create sycl::kernel_bundle object
 * in executable state for OpenCL and Level-Zero backends from SPIR-V binary.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    Dev            An opaque pointer to a sycl::device
 * @param    IL             SPIR-V binary
 * @param    Length         The size of the IL binary in bytes.
 * @param    CompileOpts    Optional compiler flags used when compiling the
 *                          SPIR-V binary.
 * @return   A new SyclKernelBundleRef pointer if the kernel_bundle creation
 * succeeded, else returns NULL.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelBundleRef
DPCTLKernelBundle_CreateFromSpirv(__dpctl_keep const DPCTLSyclContextRef Ctx,
                                  __dpctl_keep const DPCTLSyclDeviceRef Dev,
                                  __dpctl_keep const void *IL,
                                  size_t Length,
                                  const char *CompileOpts);

/*!
 * @brief Create a Sycl kernel bundle from an OpenCL kernel source string.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    Dev            An opaque pointer to a sycl::device
 * @param    Source         OpenCL source string
 * @param    CompileOpts    Extra compiler flags (refer Sycl spec.)
 * @return   A new SyclKernelBundleRef pointer if the program creation
 * succeeded, else returns NULL.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelBundleRef DPCTLKernelBundle_CreateFromOCLSource(
    __dpctl_keep const DPCTLSyclContextRef Ctx,
    __dpctl_keep const DPCTLSyclDeviceRef Dev,
    __dpctl_keep const char *Source,
    __dpctl_keep const char *CompileOpts);

/*!
 * @brief Returns the SyclKernel with given name from the program, if not found
 * then return NULL.
 *
 * @param    KBRef          Opaque pointer to a sycl::kernel_bundle
 * @param    KernelName     Name of kernel
 * @return   A SyclKernel reference if the kernel exists, else NULL
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelRef
DPCTLKernelBundle_GetKernel(__dpctl_keep DPCTLSyclKernelBundleRef KBRef,
                            __dpctl_keep const char *KernelName);

/*!
 * @brief Return True if a SyclKernel with given name exists in the program, if
 * not found then returns False.
 *
 * @param    KBRef          Opaque pointer to a sycl::kernel_bundle
 * @param    KernelName     Name of kernel
 * @return   True if the kernel exists, else False
 * @ingroup KernelBundleInterface
 */
DPCTL_API
bool DPCTLKernelBundle_HasKernel(__dpctl_keep DPCTLSyclKernelBundleRef KBRef,
                                 __dpctl_keep const char *KernelName);

/*!
 * @brief Frees the DPCTLSyclKernelBundleRef pointer.
 *
 * @param   KBRef           Opaque pointer to a sycl::kernel_bundle
 * @ingroup KernelBundleInterface
 */
DPCTL_API
void DPCTLKernelBundle_Delete(__dpctl_take DPCTLSyclKernelBundleRef KBRef);

/*!
 * @brief Returns a copy of the DPCTLSyclKernelBundleRef object.
 *
 * @param    KBRef           DPCTLSyclKernelBundleRef object to be copied.
 * @return   A new DPCTLSyclKernelBundleRef created by copying the passed in
 * DPCTLSyclKernelBundleRef object.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelBundleRef
DPCTLKernelBundle_Copy(__dpctl_keep const DPCTLSyclKernelBundleRef KBRef);

typedef struct DPCTLBuildOptionList *DPCTLBuildOptionListRef;
typedef struct DPCTLKernelNameList *DPCTLKernelNameListRef;
typedef struct DPCTLVirtualHeaderList *DPCTLVirtualHeaderListRef;
typedef struct DPCTLKernelBuildLog *DPCTLKernelBuildLogRef;

/*!
 * @brief Create an empty list of build options.
 *
 * @return Opaque pointer to the build option file list.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLBuildOptionListRef DPCTLBuildOptionList_Create();

/*!
 * @brief Frees the DPCTLBuildOptionListRef pointer.
 *
 * @param   Ref           Opaque pointer to a list of build options
 * @ingroup KernelBundleInterface
 */
DPCTL_API void
DPCTLBuildOptionList_Delete(__dpctl_take DPCTLBuildOptionListRef Ref);

/*!
 * @brief Append a build option to the list of build options
 *
 * @param Ref Opaque pointer to the list of build options
 * @param Option Option to append
 */
DPCTL_API
void DPCTLBuildOptionList_Append(__dpctl_keep DPCTLBuildOptionListRef Ref,
                                 __dpctl_keep const char *Option);

/*!
 * @brief Create an empty list of kernel names to register.
 *
 * @return Opaque pointer to the list of kernel names to register.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLKernelNameListRef DPCTLKernelNameList_Create();

/*!
 * @brief Frees the DPCTLKernelNameListRef pointer.
 *
 * @param   Ref           Opaque pointer to a list of kernels to register
 * @ingroup KernelBundleInterface
 */
DPCTL_API void
DPCTLKernelNameList_Delete(__dpctl_take DPCTLKernelNameListRef Ref);

/*!
 * @brief Append a kernel name to register to the list of build options
 *
 * @param Ref Opaque pointer to the list of kernel names
 * @param Option Kernel name to append
 */
DPCTL_API
void DPCTLKernelNameList_Append(__dpctl_keep DPCTLKernelNameListRef Ref,
                                __dpctl_keep const char *Option);
/*!
 * @brief Create an empty list of virtual header files.
 *
 * @return Opaque pointer to the virtual header file list.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLVirtualHeaderListRef DPCTLVirtualHeaderList_Create();

/*!
 * @brief Frees the DPCTLVirtualHeaderListRef pointer.
 *
 * @param   Ref           Opaque pointer to a list of virtual headers
 * @ingroup KernelBundleInterface
 */
DPCTL_API void
DPCTLVirtualHeaderList_Delete(__dpctl_take DPCTLVirtualHeaderListRef Ref);

/*!
 * @brief Append a kernel name to register to the list of virtual header files
 *
 * @param Ref Opaque pointer to the list of header files
 * @param Name Name of the virtual header file
 * @param Content Content of the virtual header
 */
DPCTL_API
void DPCTLVirtualHeaderList_Append(__dpctl_keep DPCTLVirtualHeaderListRef Ref,
                                   __dpctl_keep const char *Name,
                                   __dpctl_keep const char *Content);

/*!
 * @brief Create an empty kernel build log.
 *
 * @return Opaque pointer to the kernel build log.
 * @ingroup KernelBundleInterface
 */
DPCTL_API __dpctl_give DPCTLKernelBuildLogRef DPCTLKernelBuildLog_Create();

/*!
 * @brief Frees the DPCTLKernelBuildLogRef pointer.
 *
 * @param   Ref           Opaque pointer to a kernel build log.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
void DPCTLKernelBuildLog_Delete(__dpctl_take DPCTLKernelBuildLogRef Ref);

/*!
 * @brief Get the content of the build log.
 *
 * @param Ref   Opaque pointer to the kernel build log.
 * @return      Content of the build log
 * @ingroup     KernelBundleInterface
 */
DPCTL_API const char *
DPCTLKernelBuildLog_Get(__dpctl_keep DPCTLKernelBuildLogRef);

/*!
 * @brief Create a SYCL kernel bundle from an SYCL kernel source string.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    Dev            An opaque pointer to a sycl::device
 * @param    Source         SYCL source string
 * @param    Headers        List of virtual headers
 * @param    Names          List of kernel names to register
 * @param    CompileOpts    List of extra compiler flags (refer Sycl spec.)
 * @return   A new SyclKernelBundleRef pointer if the program creation
 * succeeded, else returns NULL.
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelBundleRef DPCTLKernelBundle_CreateFromSYCLSource(
    __dpctl_keep const DPCTLSyclContextRef Ctx,
    __dpctl_keep const DPCTLSyclDeviceRef Dev,
    __dpctl_keep const char *Source,
    __dpctl_keep DPCTLVirtualHeaderListRef Headers,
    __dpctl_keep DPCTLKernelNameListRef Names,
    __dpctl_keep DPCTLBuildOptionListRef BuildOptions,
    __dpctl_keep DPCTLKernelBuildLogRef BuildLog);

/*!
 * @brief Returns the SyclKernel with given name from the program compiled from
 * SYCL source code, if not found then return NULL.
 *
 * @param    KBRef          Opaque pointer to a sycl::kernel_bundle
 * @param    KernelName     Name of kernel
 * @return   A SyclKernel reference if the kernel exists, else NULL
 * @ingroup KernelBundleInterface
 */
DPCTL_API
__dpctl_give DPCTLSyclKernelRef
DPCTLKernelBundle_GetSyclKernel(__dpctl_keep DPCTLSyclKernelBundleRef KBRef,
                                __dpctl_keep const char *KernelName);

/*!
 * @brief Return True if a SyclKernel with given name exists in the program
 * compiled from SYCL source code, if not found then returns False.
 *
 * @param    KBRef          Opaque pointer to a sycl::kernel_bundle
 * @param    KernelName     Name of kernel
 * @return   True if the kernel exists, else False
 * @ingroup KernelBundleInterface
 */

DPCTL_API
bool DPCTLKernelBundle_HasSyclKernel(__dpctl_keep DPCTLSyclKernelBundleRef
                                         KBRef,
                                     __dpctl_keep const char *KernelName);

DPCTL_C_EXTERN_C_END
