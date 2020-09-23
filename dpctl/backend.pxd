##===------------- backend.pyx - dpctl interface ------*- Cython -*--------===##
##
##                      Data Parallel Control (dpctl)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
##
## \file
## This file defines the Cython extern types for the functions and opaque data
## types defined by dpctl's C API.
##
##===----------------------------------------------------------------------===##

# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool


cdef extern from "dppl_utils.h":
    cdef void DPPLCString_Delete (const char *str)


cdef extern from "dppl_sycl_types.h":
    cdef struct DPPLOpaqueSyclContext
    cdef struct DPPLOpaqueSyclDevice
    cdef struct DPPLOpaqueSyclKernel
    cdef struct DPPLOpaqueSyclProgram
    cdef struct DPPLOpaqueSyclQueue
    cdef struct DPPLOpaqueSyclUSM

    ctypedef DPPLOpaqueSyclContext* DPPLSyclContextRef
    ctypedef DPPLOpaqueSyclDevice*  DPPLSyclDeviceRef
    ctypedef DPPLOpaqueSyclKernel*  DPPLSyclKernelRef
    ctypedef DPPLOpaqueSyclProgram* DPPLSyclProgramRef
    ctypedef DPPLOpaqueSyclQueue*   DPPLSyclQueueRef
    ctypedef DPPLOpaqueSyclUSM*     DPPLSyclUSMRef


cdef extern from "dppl_sycl_context_interface.h":
    cdef void DPPLContext_Delete (DPPLSyclContextRef CtxtRef) except +


cdef extern from "dppl_sycl_device_interface.h":
    cdef void DPPLDevice_DumpInfo (const DPPLSyclDeviceRef DRef) except +
    cdef void DPPLDevice_Delete (DPPLSyclDeviceRef DRef) except +
    cdef void DPPLDevice_DumpInfo (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDevice_IsAccelerator (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDevice_IsCPU (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDevice_IsGPU (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDevice_IsHost (const DPPLSyclDeviceRef DRef) except +
    cdef const char* DPPLDevice_GetDriverInfo (const DPPLSyclDeviceRef DRef) \
    except +
    cdef const char* DPPLDevice_GetName (const DPPLSyclDeviceRef DRef) except +
    cdef const char* DPPLDevice_GetVendorName (const DPPLSyclDeviceRef DRef) \
    except +
    cdef bool DPPLDevice_IsHostUnifiedMemory (const DPPLSyclDeviceRef DRef) \
    except +


cdef extern from "dppl_sycl_kernel_interface.h":
    cdef const char* DPPLKernel_GetFunctionName (const DPPLSyclKernelRef KRef)
    cdef size_t DPPLKernel_GetNumArgs (const DPPLSyclKernelRef KRef)
    cdef void DPPLKernel_Delete (DPPLSyclKernelRef KRef)


cdef extern from "dppl_sycl_platform_interface.h":
    cdef size_t DPPLPlatform_GetNumPlatforms ()
    cdef void DPPLPlatform_DumpInfo ()


cdef extern from "dppl_sycl_program_interface.h":
    cdef DPPLSyclProgramRef DPPLProgram_CreateFromOCLSpirv (                   \
                                const DPPLSyclContextRef Ctx,                  \
                                const void *IL,                                \
                                size_t Length)
    cdef DPPLSyclProgramRef DPPLProgram_CreateFromOCLSource (                  \
                                const DPPLSyclContextRef Ctx,                  \
                                const char* Source,                            \
                                const char* CompileOpts)
    cdef DPPLSyclKernelRef DPPLProgram_GetKernel (DPPLSyclProgramRef PRef,     \
                                                  const char *KernelName)
    cdef bool DPPLProgram_HasKernel (DPPLSyclProgramRef PRef,                  \
                                     const char *KernelName)
    cdef void DPPLProgram_Delete (DPPLSyclProgramRef PRef)


cdef extern from "dppl_sycl_queue_interface.h":
    cdef void DPPLQueue_Delete (DPPLSyclQueueRef QRef) except +
    cdef DPPLSyclContextRef DPPLQueue_GetContext (const DPPLSyclQueueRef Q) \
         except+
    cdef DPPLSyclDeviceRef DPPLQueue_GetDevice (const DPPLSyclQueueRef Q) \
         except +


cdef extern from "dppl_sycl_queue_manager.h":
    cdef enum _device_type 'DPPLSyclDeviceType':
        _GPU 'DPPL_GPU'
        _CPU 'DPPL_CPU'

    cdef DPPLSyclQueueRef DPPLQueueMgr_GetCurrentQueue () except +
    cdef size_t DPPLQueueMgr_GetNumCPUQueues () except +
    cdef size_t DPPLQueueMgr_GetNumGPUQueues () except +
    cdef size_t DPPLQueueMgr_GetNumActivatedQueues () except +
    cdef DPPLSyclQueueRef DPPLQueueMgr_GetQueue (_device_type DTy,
                                                 size_t device_num) except +
    cdef void DPPLQueueMgr_PopQueue () except +
    cdef DPPLSyclQueueRef DPPLQueueMgr_PushQueue (_device_type DTy,
                                                  size_t device_num) except +
    cdef void DPPLQueueMgr_SetAsDefaultQueue (_device_type DTy,
                                              size_t device_num) except +


cdef extern from "dppl_sycl_usm_interface.h":
    cdef DPPLSyclUSMRef DPPLmalloc_shared (size_t size, DPPLSyclQueueRef QRef) \
         except +
    cdef DPPLSyclUSMRef DPPLmalloc_host (size_t size, DPPLSyclQueueRef QRef) \
         except +
    cdef DPPLSyclUSMRef DPPLmalloc_device (size_t size, DPPLSyclQueueRef QRef) \
         except +

    cdef void DPPLfree_with_queue (DPPLSyclUSMRef MRef,
                                   DPPLSyclQueueRef QRef) except +
    cdef void DPPLfree_with_context (DPPLSyclUSMRef MRef,
                                     DPPLSyclContextRef CRef) except +

    cdef const char* DPPLUSM_GetPointerType (DPPLSyclUSMRef MRef,
                                             DPPLSyclContextRef CRef) except +
