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
from libc.stdint cimport uint32_t


cdef extern from "dppl_utils.h":
    cdef void DPPLCString_Delete (const char *str)
    cdef void DPPLSize_t_Array_Delete (size_t* arr)

cdef extern from "dppl_sycl_enum_types.h":
    cdef enum _backend_type 'DPPLSyclBEType':
        _OPENCL          'DPPL_OPENCL'
        _HOST            'DPPL_HOST'
        _LEVEL_ZERO      'DPPL_LEVEL_ZERO'
        _CUDA            'DPPL_CUDA'
        _UNKNOWN_BACKEND 'DPPL_UNKNOWN_BACKEND'

    ctypedef _backend_type DPPLSyclBEType

    cdef enum _device_type 'DPPLSyclDeviceType':
        _GPU         'DPPL_GPU'
        _CPU         'DPPL_CPU'
        _ACCELERATOR 'DPPL_ACCELERATOR'
        _HOST_DEVICE 'DPPL_HOST_DEVICE'

    ctypedef _device_type DPPLSyclDeviceType

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


cdef extern from "dppl_sycl_device_interface.h":
    cdef void DPPLDevice_DumpInfo (const DPPLSyclDeviceRef DRef)
    cdef void DPPLDevice_Delete (DPPLSyclDeviceRef DRef)
    cdef void DPPLDevice_DumpInfo (const DPPLSyclDeviceRef DRef)
    cdef bool DPPLDevice_IsAccelerator (const DPPLSyclDeviceRef DRef)
    cdef bool DPPLDevice_IsCPU (const DPPLSyclDeviceRef DRef)
    cdef bool DPPLDevice_IsGPU (const DPPLSyclDeviceRef DRef)
    cdef bool DPPLDevice_IsHost (const DPPLSyclDeviceRef DRef)
    cpdef const char* DPPLDevice_GetDriverInfo (const DPPLSyclDeviceRef DRef)
    cpdef const char* DPPLDevice_GetName (const DPPLSyclDeviceRef DRef)
    cpdef const char* DPPLDevice_GetVendorName (const DPPLSyclDeviceRef DRef)
    cdef bool DPPLDevice_IsHostUnifiedMemory (const DPPLSyclDeviceRef DRef)
    cpdef uint32_t DPPLDevice_GetMaxComputeUnits (const DPPLSyclDeviceRef DRef)
    cpdef uint32_t DPPLDevice_GetMaxWorkItemDims (const DPPLSyclDeviceRef DRef)
    cpdef size_t* DPPLDevice_GetMaxWorkItemSizes (const DPPLSyclDeviceRef DRef)
    cpdef size_t DPPLDevice_GetMaxWorkGroupSize (const DPPLSyclDeviceRef DRef)
    cpdef uint32_t DPPLDevice_GetMaxNumSubGroups (const DPPLSyclDeviceRef DRef)
    cpdef bool DPPLDevice_HasInt64BaseAtomics (const DPPLSyclDeviceRef DRef)
    cpdef bool DPPLDevice_HasInt64ExtendedAtomics (const DPPLSyclDeviceRef DRef)


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
    cdef size_t DPPLPlatform_GetNumBackends ()
    cdef DPPLSyclBEType *DPPLPlatform_GetListOfBackends ()
    cdef void DPPLPlatform_DeleteListOfBackends (DPPLSyclBEType * BEs)


cdef extern from "dppl_sycl_context_interface.h":
    cdef bool DPPLContext_AreEq (const DPPLSyclContextRef CtxRef1,
                                 const DPPLSyclContextRef CtxRef2)
    cdef DPPLSyclBEType DPPLContext_GetBackend (const DPPLSyclContextRef CtxRef)
    cdef void DPPLContext_Delete (DPPLSyclContextRef CtxRef)


cdef extern from "dppl_sycl_program_interface.h":
    cdef DPPLSyclProgramRef DPPLProgram_CreateFromOCLSpirv (
                                const DPPLSyclContextRef Ctx,
                                const void *IL,
                                size_t Length)
    cdef DPPLSyclProgramRef DPPLProgram_CreateFromOCLSource (
                                const DPPLSyclContextRef Ctx,
                                const char* Source,
                                const char* CompileOpts)
    cdef DPPLSyclKernelRef DPPLProgram_GetKernel (DPPLSyclProgramRef PRef,
                                                  const char *KernelName)
    cdef bool DPPLProgram_HasKernel (DPPLSyclProgramRef PRef,
                                     const char *KernelName)
    cdef void DPPLProgram_Delete (DPPLSyclProgramRef PRef)


cdef extern from "dppl_sycl_queue_interface.h":
    cdef bool DPPLQueue_AreEq (const DPPLSyclQueueRef QRef1,
                               const DPPLSyclQueueRef QRef2)
    cdef void DPPLQueue_Delete (DPPLSyclQueueRef QRef)
    cdef DPPLSyclBEType DPPLQueue_GetBackend (const DPPLSyclQueueRef Q)
    cdef DPPLSyclContextRef DPPLQueue_GetContext (const DPPLSyclQueueRef Q)
    cdef DPPLSyclDeviceRef DPPLQueue_GetDevice (const DPPLSyclQueueRef Q)
    cdef DPPLSyclEventRef  DPPLQueue_SubmitRange (
                                const DPPLSyclKernelRef Ref,
                                const DPPLSyclQueueRef QRef,
                                void **Args,
                                const DPPLKernelArgType *ArgTypes,
                                size_t NArgs,
                                const size_t Range[3],
                                size_t NDims,
                                const DPPLSyclEventRef *DepEvents,
                                size_t NDepEvents)
    cdef DPPLSyclEventRef DPPLQueue_SubmitNDRange(
                                const DPPLSyclKernelRef Ref,
                                const DPPLSyclQueueRef QRef,
                                void **Args,
                                const DPPLKernelArgType *ArgTypes,
                                size_t NArgs,
                                const size_t gRange[3],
                                const size_t lRange[3],
                                size_t NDims,
                                const DPPLSyclEventRef *DepEvents,
                                size_t NDepEvents)
    cdef void DPPLQueue_Wait (const DPPLSyclQueueRef QRef)
    cdef void DPPLQueue_Memcpy (const DPPLSyclQueueRef Q,
                                void *Dest, const void *Src, size_t Count)


cdef extern from "dppl_sycl_queue_manager.h":
    cdef DPPLSyclQueueRef DPPLQueueMgr_GetCurrentQueue ()
    cdef size_t DPPLQueueMgr_GetNumQueues (DPPLSyclBEType BETy,
                                           DPPLSyclDeviceType DeviceTy)
    cdef size_t DPPLQueueMgr_GetNumActivatedQueues ()
    cdef DPPLSyclQueueRef DPPLQueueMgr_GetQueue (DPPLSyclBEType BETy,
                                                 DPPLSyclDeviceType DeviceTy,
                                                 size_t DNum)
    cdef bool DPPLQueueMgr_IsCurrentQueue (const DPPLSyclQueueRef QRef)
    cdef void DPPLQueueMgr_PopQueue ()
    cdef DPPLSyclQueueRef DPPLQueueMgr_PushQueue (DPPLSyclBEType BETy,
                                                  DPPLSyclDeviceType DeviceTy,
                                                  size_t DNum)
    cdef DPPLSyclQueueRef DPPLQueueMgr_SetAsDefaultQueue (
                              DPPLSyclBEType BETy,
                              DPPLSyclDeviceType DeviceTy,
                              size_t DNum
                          )


cdef extern from "dppl_sycl_usm_interface.h":
    cdef DPPLSyclUSMRef DPPLmalloc_shared (size_t size, DPPLSyclQueueRef QRef)
    cdef DPPLSyclUSMRef DPPLmalloc_host (size_t size, DPPLSyclQueueRef QRef)
    cdef DPPLSyclUSMRef DPPLmalloc_device (size_t size, DPPLSyclQueueRef QRef)
    cdef void DPPLfree_with_queue (DPPLSyclUSMRef MRef,
                                   DPPLSyclQueueRef QRef)
    cdef void DPPLfree_with_context (DPPLSyclUSMRef MRef,
                                     DPPLSyclContextRef CRef)
    cdef const char* DPPLUSM_GetPointerType (DPPLSyclUSMRef MRef,
                                             DPPLSyclContextRef CRef)
