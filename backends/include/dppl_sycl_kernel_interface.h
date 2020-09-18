//===---- dppl_sycl_kernel_interface.h - DPPL-SYCL interface --*--C++ --*--===//
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
/// This header declares a C API to create Sycl kernels from OpenCL kernels. In
/// future, API to create interoperability kernels from other languages such as
/// Level-0 driver API may be added here.
///
/// \todo Investigate what we should do when we add support for Level-0 API.
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
 * @brief Enum of currently support types for kernel arguments.
 *
 * \todo Add support for sycl::buffer
 *
 */
enum DPPLArgTypes
{
    CHAR,
    SIGNED_CHAR,
    UNSIGNED_CHAR,
    SHORT,
    INT,
    UNSIGNED_INT,
    LONG,
    UNSIGNED_LONG,
    LONG_LONG,
    UNSIGNED_LONG_LONG,
    SIZE_T,
    FLOAT,
    DOUBLE,
    LONG_DOUBLE,
    CHAR_P,
    SIGNED_CHAR_P,
    UNSIGNED_CHAR_P,
    SHORT_P,
    INT_P,
    UNSIGNED_INT_P,
    LONG_P,
    UNSIGNED_LONG_P,
    LONG_LONG_P,
    UNSIGNED_LONG_LONG_P,
    SIZE_T_P,
    FLOAT_P,
    DOUBLE_P,
    LONG_DOUBLE_P
};

/*!
 * @brief A union representing an OpenCL kernel argument that is either a basic
 * C type of a pointer of the type.
 *
 */
union DPPLArgValue
{
    char                char_arg;
    signed char         schar_arg;
    unsigned char       uchar_arg;
    short               short_arg;
    int                 int_arg;
    unsigned int        uint_arg;
    long                long_arg;
    unsigned long       ulong_arg;
    long long           longlong_arg;
    unsigned long long  ulonglong_arg;
    size_t              size_t_arg;
    float               float_arg;
    double              double_arg;
    long double         longdouble_arg;
    char                *char_p_arg;
    signed char         *schar_p_arg;
    unsigned char       *uchar_p_arg;
    short               *short_p_arg;
    int                 *int_p_arg;
    unsigned int        *uint_p_arg;
    long                *long_p_arg;
    unsigned long       *ulong_p_arg;
    long long           *longlong_p_arg;
    unsigned long long  *ulonglong_p_arg;
    size_t              *size_t_p_arg;
    float               *float_p_arg;
    double              *double_p_arg;
    long double         *longdouble_p_arg;
};

/*!
 * @brief The tagged union is used to pass through OpenCL kernel arguments to
 * Sycl.
 *
 */
struct DPPLKernelArg
{
    enum DPPLArgTypes argType;
    union DPPLArgValue argVal;
};

/*!
 * @brief Create a Sycl Kernel from an OpenCL SPIR-V binary
 *
 * Sycl 1.2 does expose any method to create a sycl::program from a SPIR-V IL
 * file. To get around this limitation, we need to use the Sycl feature to
 * create an interoperability kernel from an OpenCL kernel. This function first
 * creates an OpenCL program and kernel from the SPIR-V binary and then using
 * the Sycl-OpenCL interoperability feature creates a Sycl kernel from the
 * OpenCL kernel.
 *
 * The feature to create a Sycl kernel from a SPIR-V IL binary will be available
 * in Sycl 2.0.
 *
 * @param    Ctx            An opaque pointer to a sycl::context
 * @param    IL             SPIR-V binary
 * @return   A new SyclProgramRef pointer if the program creation succeeded,
 *           else returns NULL.
 */
DPPL_API
__dppl_give DPPLSyclKernelRef
DPPLKernel_CreateKernelFromSpirv (__dppl_keep const DPPLSyclContextRef Ctx,
                                  __dppl_keep const void *IL,
                                  size_t length,
                                  const char *KernelName = nullptr);

/*!
 * @brief Returns a C string for the kernel name.
 *
 * @param    KRef           DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   If a kernel name exists then returns it as a C string, else
 *           returns a nullptr.
 */
DPPL_API
__dppl_give const char*
DPPLKernel_GetFunctionName (__dppl_keep const DPPLSyclKernelRef KRef);

/*!
 * @brief Returns the number of arguments for the OpenCL kernel.
 *
 * @param    KRef           DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 * @return   Returns the number of arguments for the OpenCL interoperability
 *           kernel.
 */
DPPL_API
size_t
DPPLKernel_GetNumArgs (__dppl_keep const DPPLSyclKernelRef KRef);

/*!
 * @brief Deletes the DPPLSyclKernelRef after casting it to a sycl::kernel.
 *
 * @param    KRef           DPPLSyclKernelRef pointer to an OpenCL
 *                          interoperability kernel.
 */
DPPL_API
void
DPPLKernel_DeleteKernelRef (__dppl_take DPPLSyclKernelRef KRef);


/*!
 * @brief Submits the kernel to the specified queue using give arguments.
 *
 * A wrapper over sycl::queue.submit(). The function takes an OpenCL
 * interoperability kernel, the kernel arguments, and a sycl queue as input
 * arguments. The kernel arguments are passed in as an array of the
 * DPPLKernelArg tagged union.
 *
 * \todo sycl::buffer arguments are not supported yet.
 *
 * @param    KRef           Opaque pointer to a OpenCL interoperability kernel
 *                          wrapped inside a sycl::kernel.
 * @param    QRef           Opaque pointer to the sycl::queue where the kernel
 *                          will be enqueued.
 * @param    Args           An array of the DPPLKernelArg tagged union type that
 *                          represents the kernel arguments for the kernel.
 * @param    NArgs          The number of kernel arguments (size of Args array).
 * @param    Range          Array storing the range dimensions that can have a
 *                          maximum size of three. Note the number of values
 *                          in the array depends on the number of dimensions.
 * @param    NDims          Number of dimensions in the range (size of Range).
 * @return   A opaque pointer to the sycl::event returned by the
 *           sycl::queue.submit() function.
 */
DPPL_API
DPPLSyclEventRef
DPPLKernel_Submit (__dppl_keep DPPLSyclKernelRef KRef,
                   __dppl_keep DPPLSyclQueueRef QRef,
                   __dppl_keep DPPLKernelArg *Args,
                   size_t NArgs,
                   size_t Range[3],
                   size_t NDims);

DPPL_C_EXTERN_C_END
