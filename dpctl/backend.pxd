##===------------- backend.pyx - dpctl module -------*- Cython -*----------===##
##
##                      Data Parallel Control (dpCtl)
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
    cdef struct DPPLOpaqueSyclEvent
    cdef struct DPPLOpaqueSyclKernel
    cdef struct DPPLOpaqueSyclProgram
    cdef struct DPPLOpaqueSyclQueue
    cdef struct DPPLOpaqueSyclUSM

    ctypedef DPPLOpaqueSyclContext* DPPLSyclContextRef
    ctypedef DPPLOpaqueSyclDevice*  DPPLSyclDeviceRef
    ctypedef DPPLOpaqueSyclEvent*   DPPLSyclEventRef
    ctypedef DPPLOpaqueSyclKernel*  DPPLSyclKernelRef
    ctypedef DPPLOpaqueSyclProgram* DPPLSyclProgramRef
    ctypedef DPPLOpaqueSyclQueue*   DPPLSyclQueueRef
    ctypedef DPPLOpaqueSyclUSM*     DPPLSyclUSMRef


cdef extern from "dppl_sycl_context_interface.h":
    cdef void DPPLContext_Delete (DPPLSyclContextRef CtxtRef) except +


cdef extern from "dppl_sycl_device_interface.h":
    cdef enum _device_type 'DPPLSyclDeviceType':
        _GPU 'DPPL_GPU'
        _CPU 'DPPL_CPU'
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


cdef extern from "dppl_sycl_event_interface.h":
    cdef void DPPLEvent_Wait (DPPLSyclEventRef ERef)
    cdef void DPPLEvent_Delete (DPPLSyclEventRef ERef)


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
    cdef enum _arg_data_type 'DPPLKernelArgType':
        _CHAR               'DPPL_CHAR',
        _SIGNED_CHAR        'DPPL_SIGNED_CHAR',
        _UNSIGNED_CHAR      'DPPL_UNSIGNED_CHAR',
        _SHORT              'DPPL_SHORT',
        _INT                'DPPL_INT',
        _UNSIGNED_INT       'DPPL_INT',
        _LONG               'DPPL_LONG',
        _UNSIGNED_LONG      'DPPL_UNSIGNED_LONG',
        _LONG_LONG          'DPPL_LONG_LONG',
        _UNSIGNED_LONG_LONG 'DPPL_UNSIGNED_LONG_LONG',
        _SIZE_T             'DPPL_SIZE_T',
        _FLOAT              'DPPL_FLOAT',
        _DOUBLE             'DPPL_DOUBLE',
        _LONG_DOUBLE        'DPPL_DOUBLE',
        _VOID_PTR           'DPPL_VOID_PTR'
    ctypedef _arg_data_type DPPLKernelArgType
    cdef void DPPLQueue_Delete (DPPLSyclQueueRef QRef)
    cdef DPPLSyclContextRef DPPLQueue_GetContext (const DPPLSyclQueueRef Q)
    cdef DPPLSyclDeviceRef DPPLQueue_GetDevice (const DPPLSyclQueueRef Q)
    cdef DPPLSyclEventRef  DPPLQueue_SubmitRange (                             \
                                const DPPLSyclKernelRef Ref,                   \
                                const DPPLSyclQueueRef QRef,                   \
                                void **Args,                                   \
                                const DPPLKernelArgType *ArgTypes,             \
                                size_t NArgs,                                  \
                                const size_t Range[3],                         \
                                size_t NDims,                                  \
                                const DPPLSyclEventRef *DepEvents,             \
                                size_t NDepEvents)
    cdef DPPLSyclEventRef DPPLQueue_SubmitNDRange(                             \
                                const DPPLSyclKernelRef Ref,                   \
                                const DPPLSyclQueueRef QRef,                   \
                                void **Args,                                   \
                                const DPPLKernelArgType *ArgTypes,             \
                                size_t NArgs,                                  \
                                const size_t gRange[3],                        \
                                const size_t lRange[3],                        \
                                size_t NDims,                                  \
                                const DPPLSyclEventRef *DepEvents,             \
                                size_t NDepEvents)
    cdef void DPPLQueue_Wait (const DPPLSyclQueueRef QRef)
    cdef void DPPLQueue_memcpy (const DPPLSyclQueueRef Q,
                                void *Dest, const void *Src, size_t Count)


cdef extern from "dppl_sycl_queue_manager.h":
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
