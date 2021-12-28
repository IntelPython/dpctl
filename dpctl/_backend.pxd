#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# distutils: language = c++
# cython: language_level=3

"""This file defines the Cython extern types for the functions and opaque data
types defined by dpctl's C API.
"""

from libc.stdint cimport int64_t, uint32_t
from libcpp cimport bool


cdef extern from "syclinterface/dpctl_error_handler_type.h":
    ctypedef void error_handler_callback(int err_code)

cdef extern from "syclinterface/dpctl_utils.h":
    cdef void DPCTLCString_Delete(const char *str)
    cdef void DPCTLSize_t_Array_Delete(size_t *arr)


cdef extern from "syclinterface/dpctl_sycl_enum_types.h":
    ctypedef enum _backend_type 'DPCTLSyclBackendType':
        _ALL_BACKENDS    'DPCTL_ALL_BACKENDS'
        _CUDA            'DPCTL_CUDA'
        _HOST            'DPCTL_HOST'
        _LEVEL_ZERO      'DPCTL_LEVEL_ZERO'
        _OPENCL          'DPCTL_OPENCL'
        _UNKNOWN_BACKEND 'DPCTL_UNKNOWN_BACKEND'

    ctypedef enum _device_type 'DPCTLSyclDeviceType':
        _ACCELERATOR    'DPCTL_ACCELERATOR'
        _ALL_DEVICES    'DPCTL_ALL'
        _AUTOMATIC      'DPCTL_AUTOMATIC'
        _CPU            'DPCTL_CPU'
        _CUSTOM         'DPCTL_CUSTOM'
        _GPU            'DPCTL_GPU'
        _HOST_DEVICE    'DPCTL_HOST_DEVICE'
        _UNKNOWN_DEVICE 'DPCTL_UNKNOWN_DEVICE'

    ctypedef enum _arg_data_type 'DPCTLKernelArgType':
        _CHAR               'DPCTL_CHAR',
        _SIGNED_CHAR        'DPCTL_SIGNED_CHAR',
        _UNSIGNED_CHAR      'DPCTL_UNSIGNED_CHAR',
        _SHORT              'DPCTL_SHORT',
        _INT                'DPCTL_INT',
        _UNSIGNED_INT       'DPCTL_UNSIGNED_INT',
        _UNSIGNED_INT8      'DPCTL_UNSIGNED_INT8',
        _LONG               'DPCTL_LONG',
        _UNSIGNED_LONG      'DPCTL_UNSIGNED_LONG',
        _LONG_LONG          'DPCTL_LONG_LONG',
        _UNSIGNED_LONG_LONG 'DPCTL_UNSIGNED_LONG_LONG',
        _SIZE_T             'DPCTL_SIZE_T',
        _FLOAT              'DPCTL_FLOAT',
        _DOUBLE             'DPCTL_DOUBLE',
        _LONG_DOUBLE        'DPCTL_DOUBLE',
        _VOID_PTR           'DPCTL_VOID_PTR'

    ctypedef enum _queue_property_type 'DPCTLQueuePropertyType':
        _DEFAULT_PROPERTY   'DPCTL_DEFAULT_PROPERTY'
        _ENABLE_PROFILING   'DPCTL_ENABLE_PROFILING'
        _IN_ORDER           'DPCTL_IN_ORDER'

    ctypedef enum _aspect_type 'DPCTLSyclAspectType':
        _host                               'host',
        _cpu                                'cpu',
        _gpu                                'gpu',
        _accelerator                        'accelerator',
        _custom                             'custom',
        _fp16                               'fp16',
        _fp64                               'fp64',
        _int64_base_atomics                 'int64_base_atomics',
        _int64_extended_atomics             'int64_extended_atomics',
        _image                              'image',
        _online_compiler                    'online_compiler',
        _online_linker                      'online_linker',
        _queue_profiling                    'queue_profiling',
        _usm_device_allocations             'usm_device_allocations',
        _usm_host_allocations               'usm_host_allocations',
        _usm_shared_allocations             'usm_shared_allocations',
        _usm_restricted_shared_allocations  'usm_restricted_shared_allocations',
        _usm_system_allocator               'usm_system_allocator'

    ctypedef enum _partition_affinity_domain_type 'DPCTLPartitionAffinityDomainType':
        _not_applicable                     'not_applicable',
        _numa                               'numa',
        _L4_cache                           'L4_cache',
        _L3_cache                           'L3_cache',
        _L2_cache                           'L2_cache',
        _L1_cache                           'L1_cache',
        _next_partitionable                 'next_partitionable',

    ctypedef enum _event_status_type 'DPCTLSyclEventStatusType':
        _UNKNOWN_STATUS     'DPCTL_UNKNOWN_STATUS'
        _SUBMITTED          'DPCTL_SUBMITTED'
        _RUNNING            'DPCTL_RUNNING'
        _COMPLETE           'DPCTL_COMPLETE'


cdef extern from "syclinterface/dpctl_sycl_types.h":
    cdef struct DPCTLOpaqueSyclContext
    cdef struct DPCTLOpaqueSyclDevice
    cdef struct DPCTLOpaqueSyclDeviceSelector
    cdef struct DPCTLOpaqueSyclEvent
    cdef struct DPCTLOpaqueSyclKernel
    cdef struct DPCTLOpaqueSyclPlatform
    cdef struct DPCTLOpaqueSyclProgram
    cdef struct DPCTLOpaqueSyclQueue
    cdef struct DPCTLOpaqueSyclUSM

    ctypedef DPCTLOpaqueSyclContext        *DPCTLSyclContextRef
    ctypedef DPCTLOpaqueSyclDevice         *DPCTLSyclDeviceRef
    ctypedef DPCTLOpaqueSyclDeviceSelector *DPCTLSyclDeviceSelectorRef
    ctypedef DPCTLOpaqueSyclEvent          *DPCTLSyclEventRef
    ctypedef DPCTLOpaqueSyclKernel         *DPCTLSyclKernelRef
    ctypedef DPCTLOpaqueSyclPlatform       *DPCTLSyclPlatformRef
    ctypedef DPCTLOpaqueSyclProgram        *DPCTLSyclProgramRef
    ctypedef DPCTLOpaqueSyclQueue          *DPCTLSyclQueueRef
    ctypedef DPCTLOpaqueSyclUSM            *DPCTLSyclUSMRef


cdef extern from "syclinterface/dpctl_sycl_device_manager.h":
    cdef struct DPCTLDeviceVector
    ctypedef DPCTLDeviceVector *DPCTLDeviceVectorRef


cdef extern from "syclinterface/dpctl_sycl_device_interface.h":
    cdef bool DPCTLDevice_AreEq(const DPCTLSyclDeviceRef DRef1,
                                const DPCTLSyclDeviceRef DRef2)
    cdef DPCTLSyclDeviceRef DPCTLDevice_Copy(const DPCTLSyclDeviceRef DRef)
    cdef DPCTLSyclDeviceRef DPCTLDevice_Create()
    cdef DPCTLSyclDeviceRef DPCTLDevice_CreateFromSelector(
        const DPCTLSyclDeviceSelectorRef DSRef)
    cdef void DPCTLDevice_Delete(DPCTLSyclDeviceRef DRef)
    cdef _backend_type DPCTLDevice_GetBackend(const DPCTLSyclDeviceRef)
    cdef _device_type DPCTLDevice_GetDeviceType(const DPCTLSyclDeviceRef)
    cdef const char *DPCTLDevice_GetDriverVersion(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetGlobalMemSize(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetLocalMemSize(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetMaxComputeUnits(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetMaxNumSubGroups(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetMaxWorkGroupSize(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetMaxWorkItemDims(const DPCTLSyclDeviceRef DRef)
    cdef size_t *DPCTLDevice_GetMaxWorkItemSizes(const DPCTLSyclDeviceRef DRef)
    cdef const char *DPCTLDevice_GetName(const DPCTLSyclDeviceRef DRef)
    cdef DPCTLSyclPlatformRef DPCTLDevice_GetPlatform(
        const DPCTLSyclDeviceRef DRef)
    cdef const char *DPCTLDevice_GetVendor(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_Hash(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsAccelerator(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsCPU(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsGPU(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsHost(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsHostUnifiedMemory(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_GetSubGroupIndependentForwardProgress(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthChar(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthShort(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthInt(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthLong(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthFloat(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthDouble(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthHalf(const DPCTLSyclDeviceRef DRef)
    cpdef bool DPCTLDevice_HasAspect(const DPCTLSyclDeviceRef, _aspect_type)
    cdef uint32_t DPCTLDevice_GetMaxReadImageArgs(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetMaxWriteImageArgs(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetImage2dMaxWidth(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetImage2dMaxHeight(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetImage3dMaxWidth(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetImage3dMaxHeight(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_GetImage3dMaxDepth(const DPCTLSyclDeviceRef DRef)
    cdef DPCTLDeviceVectorRef DPCTLDevice_CreateSubDevicesEqually(
        const DPCTLSyclDeviceRef DRef, size_t count)
    cdef DPCTLDeviceVectorRef DPCTLDevice_CreateSubDevicesByCounts(
        const DPCTLSyclDeviceRef DRef, size_t *counts, size_t ncounts)
    cdef DPCTLDeviceVectorRef DPCTLDevice_CreateSubDevicesByAffinity(
        const DPCTLSyclDeviceRef DRef,
        _partition_affinity_domain_type PartitionAffinityDomainTy)
    cdef DPCTLSyclDeviceRef DPCTLDevice_GetParentDevice(const DPCTLSyclDeviceRef DRef)


cdef extern from "syclinterface/dpctl_sycl_device_manager.h":
    cdef DPCTLDeviceVectorRef DPCTLDeviceVector_CreateFromArray(
        size_t nelems,
        DPCTLSyclDeviceRef *elems)
    cdef void DPCTLDeviceVector_Delete(DPCTLDeviceVectorRef DVRef)
    cdef void DPCTLDeviceVector_Clear(DPCTLDeviceVectorRef DVRef)
    cdef size_t DPCTLDeviceVector_Size(DPCTLDeviceVectorRef DVRef)
    cdef DPCTLSyclDeviceRef DPCTLDeviceVector_GetAt(
        DPCTLDeviceVectorRef DVRef,
        size_t index)
    cdef DPCTLDeviceVectorRef DPCTLDeviceMgr_GetDevices(int device_identifier)
    cdef int DPCTLDeviceMgr_GetPositionInDevices(
        const DPCTLSyclDeviceRef DRef,
        int device_identifier)
    cdef size_t DPCTLDeviceMgr_GetNumDevices(int device_identifier)
    cdef const char * DPCTLDeviceMgr_GetDeviceInfoStr(const DPCTLSyclDeviceRef DRef)
    cdef DPCTLSyclContextRef DPCTLDeviceMgr_GetCachedContext(
        const DPCTLSyclDeviceRef DRef)
    cdef int64_t DPCTLDeviceMgr_GetRelativeId(const DPCTLSyclDeviceRef DRef)


cdef extern from "syclinterface/dpctl_sycl_device_selector_interface.h":
    DPCTLSyclDeviceSelectorRef DPCTLAcceleratorSelector_Create()
    DPCTLSyclDeviceSelectorRef DPCTLDefaultSelector_Create()
    DPCTLSyclDeviceSelectorRef DPCTLCPUSelector_Create()
    DPCTLSyclDeviceSelectorRef DPCTLFilterSelector_Create(const char *)
    DPCTLSyclDeviceSelectorRef DPCTLGPUSelector_Create()
    DPCTLSyclDeviceSelectorRef DPCTLHostSelector_Create()
    void DPCTLDeviceSelector_Delete(DPCTLSyclDeviceSelectorRef)
    int DPCTLDeviceSelector_Score(DPCTLSyclDeviceSelectorRef, DPCTLSyclDeviceRef)


cdef extern from "syclinterface/dpctl_sycl_event_interface.h":
    cdef DPCTLSyclEventRef DPCTLEvent_Create()
    cdef DPCTLSyclEventRef DPCTLEvent_Copy(const DPCTLSyclEventRef ERef)
    cdef void DPCTLEvent_Wait(DPCTLSyclEventRef ERef) nogil
    cdef void DPCTLEvent_WaitAndThrow(DPCTLSyclEventRef ERef) nogil
    cdef void DPCTLEvent_Delete(DPCTLSyclEventRef ERef)
    cdef _event_status_type DPCTLEvent_GetCommandExecutionStatus(DPCTLSyclEventRef ERef)
    cdef _backend_type DPCTLEvent_GetBackend(DPCTLSyclEventRef ERef)
    cdef struct DPCTLEventVector
    ctypedef DPCTLEventVector *DPCTLEventVectorRef
    cdef void DPCTLEventVector_Delete(DPCTLEventVectorRef EVRef)
    cdef size_t DPCTLEventVector_Size(DPCTLEventVectorRef EVRef)
    cdef DPCTLSyclEventRef DPCTLEventVector_GetAt(
        DPCTLEventVectorRef EVRef,
        size_t index)
    cdef DPCTLEventVectorRef DPCTLEvent_GetWaitList(
        DPCTLSyclEventRef ERef)
    cdef size_t DPCTLEvent_GetProfilingInfoSubmit(DPCTLSyclEventRef ERef)
    cdef size_t DPCTLEvent_GetProfilingInfoStart(DPCTLSyclEventRef ERef)
    cdef size_t DPCTLEvent_GetProfilingInfoEnd(DPCTLSyclEventRef ERef)


cdef extern from "syclinterface/dpctl_sycl_kernel_interface.h":
    cdef const char* DPCTLKernel_GetFunctionName(const DPCTLSyclKernelRef KRef)
    cdef size_t DPCTLKernel_GetNumArgs(const DPCTLSyclKernelRef KRef)
    cdef void DPCTLKernel_Delete(DPCTLSyclKernelRef KRef)


cdef extern from "syclinterface/dpctl_sycl_platform_manager.h":
    cdef struct DPCTLPlatformVector
    ctypedef DPCTLPlatformVector *DPCTLPlatformVectorRef

    cdef void DPCTLPlatformVector_Delete(DPCTLPlatformVectorRef)
    cdef void DPCTLPlatformVector_Clear(DPCTLPlatformVectorRef)
    cdef size_t DPCTLPlatformVector_Size(DPCTLPlatformVectorRef)
    cdef DPCTLSyclPlatformRef DPCTLPlatformVector_GetAt(
        DPCTLPlatformVectorRef,
        size_t index)
    cdef void DPCTLPlatformMgr_PrintInfo(const DPCTLSyclPlatformRef, size_t)


cdef extern from "syclinterface/dpctl_sycl_platform_interface.h":
    cdef DPCTLSyclPlatformRef DPCTLPlatform_Copy(const DPCTLSyclPlatformRef)
    cdef DPCTLSyclPlatformRef DPCTLPlatform_Create()
    cdef DPCTLSyclPlatformRef DPCTLPlatform_CreateFromSelector(
        const DPCTLSyclDeviceSelectorRef)
    cdef void DPCTLPlatform_Delete(DPCTLSyclPlatformRef)
    cdef _backend_type DPCTLPlatform_GetBackend(const DPCTLSyclPlatformRef)
    cdef const char *DPCTLPlatform_GetName(const DPCTLSyclPlatformRef)
    cdef const char *DPCTLPlatform_GetVendor(const DPCTLSyclPlatformRef)
    cdef const char *DPCTLPlatform_GetVersion(const DPCTLSyclPlatformRef)
    cdef DPCTLPlatformVectorRef DPCTLPlatform_GetPlatforms()


cdef extern from "syclinterface/dpctl_sycl_context_interface.h":
    cdef DPCTLSyclContextRef DPCTLContext_Create(
        const DPCTLSyclDeviceRef DRef,
        error_handler_callback *handler,
        int properties)
    cdef DPCTLSyclContextRef DPCTLContext_CreateFromDevices(
        const DPCTLDeviceVectorRef DVRef,
        error_handler_callback *handler,
        int properties)
    cdef DPCTLSyclContextRef DPCTLContext_Copy(
        const DPCTLSyclContextRef CRef)
    cdef DPCTLDeviceVectorRef DPCTLContext_GetDevices(
        const DPCTLSyclContextRef CRef)
    cdef size_t DPCTLContext_DeviceCount(const DPCTLSyclContextRef CRef)
    cdef bool DPCTLContext_AreEq(const DPCTLSyclContextRef CtxRef1,
                                 const DPCTLSyclContextRef CtxRef2)
    cdef size_t DPCTLContext_Hash(const DPCTLSyclContextRef CRef)
    cdef _backend_type DPCTLContext_GetBackend(const DPCTLSyclContextRef)
    cdef void DPCTLContext_Delete(DPCTLSyclContextRef CtxRef)


cdef extern from "syclinterface/dpctl_sycl_program_interface.h":
    cdef DPCTLSyclProgramRef DPCTLProgram_CreateFromSpirv(
        const DPCTLSyclContextRef Ctx,
        const void *IL,
        size_t Length,
        const char *CompileOpts)
    cdef DPCTLSyclProgramRef DPCTLProgram_CreateFromOCLSource(
        const DPCTLSyclContextRef Ctx,
        const char *Source,
        const char *CompileOpts)
    cdef DPCTLSyclKernelRef DPCTLProgram_GetKernel(
        DPCTLSyclProgramRef PRef,
        const char *KernelName)
    cdef bool DPCTLProgram_HasKernel(DPCTLSyclProgramRef PRef,
                                     const char *KernelName)
    cdef void DPCTLProgram_Delete(DPCTLSyclProgramRef PRef)


cdef extern from "syclinterface/dpctl_sycl_queue_interface.h":
    cdef bool DPCTLQueue_AreEq(const DPCTLSyclQueueRef QRef1,
                               const DPCTLSyclQueueRef QRef2)
    cdef DPCTLSyclQueueRef DPCTLQueue_Create(
        const DPCTLSyclContextRef CRef,
        const DPCTLSyclDeviceRef DRef,
        error_handler_callback *handler,
        int properties)
    cdef DPCTLSyclQueueRef DPCTLQueue_CreateForDevice(
        const DPCTLSyclDeviceRef dRef,
        error_handler_callback *handler,
        int properties)
    cdef void DPCTLQueue_Delete(DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclQueueRef DPCTLQueue_Copy(DPCTLSyclQueueRef QRef)
    cdef _backend_type DPCTLQueue_GetBackend(const DPCTLSyclQueueRef Q)
    cdef DPCTLSyclContextRef DPCTLQueue_GetContext(const DPCTLSyclQueueRef Q)
    cdef DPCTLSyclDeviceRef DPCTLQueue_GetDevice(const DPCTLSyclQueueRef Q)
    cdef size_t DPCTLQueue_Hash(const DPCTLSyclQueueRef Q)
    cdef DPCTLSyclEventRef  DPCTLQueue_SubmitRange(
        const DPCTLSyclKernelRef Ref,
        const DPCTLSyclQueueRef QRef,
        void **Args,
        const _arg_data_type *ArgTypes,
        size_t NArgs,
        const size_t Range[3],
        size_t NDims,
        const DPCTLSyclEventRef *DepEvents,
        size_t NDepEvents)
    cdef DPCTLSyclEventRef DPCTLQueue_SubmitNDRange(
        const DPCTLSyclKernelRef Ref,
        const DPCTLSyclQueueRef QRef,
        void **Args,
        const _arg_data_type *ArgTypes,
        size_t NArgs,
        const size_t gRange[3],
        const size_t lRange[3],
        size_t NDims,
        const DPCTLSyclEventRef *DepEvents,
        size_t NDepEvents)
    cdef void DPCTLQueue_Wait(const DPCTLSyclQueueRef QRef) nogil
    cdef DPCTLSyclEventRef DPCTLQueue_Memcpy(
        const DPCTLSyclQueueRef Q,
        void *Dest,
        const void *Src,
        size_t Count)
    cdef DPCTLSyclEventRef DPCTLQueue_Prefetch(
        const DPCTLSyclQueueRef Q,
        const void *Src,
        size_t Count)
    cdef DPCTLSyclEventRef DPCTLQueue_MemAdvise(
        const DPCTLSyclQueueRef Q,
        const void *Src,
        size_t Count,
        int Advice)
    cdef bool DPCTLQueue_IsInOrder(const DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclEventRef DPCTLQueue_SubmitBarrier(
        const DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclEventRef DPCTLQueue_SubmitBarrierForEvents(
        const DPCTLSyclQueueRef QRef,
        const DPCTLSyclEventRef *DepEvents,
        size_t NDepEvents)
    cdef bool DPCTLQueue_HasEnableProfiling(const DPCTLSyclQueueRef QRef)


cdef extern from "syclinterface/dpctl_sycl_queue_manager.h":
    cdef DPCTLSyclQueueRef DPCTLQueueMgr_GetCurrentQueue()
    cdef bool DPCTLQueueMgr_GlobalQueueIsCurrent()
    cdef bool DPCTLQueueMgr_IsCurrentQueue(const DPCTLSyclQueueRef QRef)
    cdef void DPCTLQueueMgr_PopQueue()
    cdef void DPCTLQueueMgr_PushQueue(const DPCTLSyclQueueRef dRef)
    cdef void DPCTLQueueMgr_SetGlobalQueue(const DPCTLSyclQueueRef dRef)
    cdef size_t DPCTLQueueMgr_GetQueueStackSize()


cdef extern from "syclinterface/dpctl_sycl_usm_interface.h":
    cdef DPCTLSyclUSMRef DPCTLmalloc_shared(
        size_t size,
        DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclUSMRef DPCTLmalloc_host(
        size_t size,
        DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclUSMRef DPCTLmalloc_device(size_t size, DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclUSMRef DPCTLaligned_alloc_shared(
        size_t alignment,
        size_t size,
        DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclUSMRef DPCTLaligned_alloc_host(
        size_t alignment,
        size_t size,
        DPCTLSyclQueueRef QRef)
    cdef DPCTLSyclUSMRef DPCTLaligned_alloc_device(
        size_t alignment,
        size_t size,
        DPCTLSyclQueueRef QRef)
    cdef void DPCTLfree_with_queue(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclQueueRef QRef)
    cdef void DPCTLfree_with_context(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclContextRef CRef)
    cdef const char* DPCTLUSM_GetPointerType(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclContextRef CRef)
    cdef DPCTLSyclDeviceRef DPCTLUSM_GetPointerDevice(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclContextRef CRef)
