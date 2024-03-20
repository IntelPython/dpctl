#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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

from libc.stdint cimport int64_t, uint32_t, uint64_t
from libcpp cimport bool


cdef extern from "syclinterface/dpctl_error_handler_type.h":
    ctypedef void error_handler_callback(int err_code)

cdef extern from "syclinterface/dpctl_utils.h":
    cdef void DPCTLCString_Delete(const char *str)
    cdef void DPCTLSize_t_Array_Delete(size_t *arr)


cdef extern from "syclinterface/dpctl_sycl_enum_types.h":
    ctypedef enum _usm_type 'DPCTLSyclUSMType':
        _USM_UNKNOWN     'DPCTL_USM_UNKNOWN'
        _USM_DEVICE      'DPCTL_USM_DEVICE'
        _USM_SHARED      'DPCTL_USM_SHARED'
        _USM_HOST        'DPCTL_USM_HOST'

    ctypedef enum _backend_type 'DPCTLSyclBackendType':
        _ALL_BACKENDS    'DPCTL_ALL_BACKENDS'
        _CUDA            'DPCTL_CUDA'
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
        _UNKNOWN_DEVICE 'DPCTL_UNKNOWN_DEVICE'

    ctypedef enum _arg_data_type 'DPCTLKernelArgType':
        _INT8_T             'DPCTL_INT8_T',
        _UINT8_T            'DPCTL_UINT8_T',
        _INT16_T            'DPCTL_INT16_T',
        _UINT16_T           'DPCTL_UINT16_T',
        _INT32_T            'DPCTL_INT32_T',
        _UINT32_T           'DPCTL_UINT32_T',
        _INT64_T            'DPCTL_INT64_T',
        _UINT64_T           'DPCTL_UINT64_T',
        _FLOAT              'DPCTL_FLOAT32_T',
        _DOUBLE             'DPCTL_FLOAT64_T',
        _VOID_PTR           'DPCTL_VOID_PTR',
        _LOCAL_ACCESSOR     'DPCTL_LOCAL_ACCESSOR'

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
        _atomic64                           'atomic64',
        _image                              'image',
        _online_compiler                    'online_compiler',
        _online_linker                      'online_linker',
        _queue_profiling                    'queue_profiling',
        _usm_device_allocations             'usm_device_allocations',
        _usm_host_allocations               'usm_host_allocations',
        _usm_shared_allocations             'usm_shared_allocations',
        _usm_system_allocations             'usm_system_allocations',
        _usm_atomic_host_allocations        'usm_atomic_host_allocations',
        _usm_atomic_shared_allocations      'usm_atomic_shared_allocations',
        _host_debuggable                    'host_debuggable',

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

    ctypedef enum _global_mem_cache_type 'DPCTLGlobalMemCacheType':
        _MEM_CACHE_TYPE_INDETERMINATE    'DPCTL_MEM_CACHE_TYPE_INDETERMINATE'
        _MEM_CACHE_TYPE_NONE             'DPCTL_MEM_CACHE_TYPE_NONE'
        _MEM_CACHE_TYPE_READ_ONLY        'DPCTL_MEM_CACHE_TYPE_READ_ONLY'
        _MEM_CACHE_TYPE_READ_WRITE       'DPCTL_MEM_CACHE_TYPE_READ_WRITE'


cdef extern from "syclinterface/dpctl_sycl_types.h":
    cdef struct DPCTLOpaqueSyclContext
    cdef struct DPCTLOpaqueSyclDevice
    cdef struct DPCTLOpaqueSyclDeviceSelector
    cdef struct DPCTLOpaqueSyclEvent
    cdef struct DPCTLOpaqueSyclKernel
    cdef struct DPCTLOpaqueSyclPlatform
    cdef struct DPCTLOpaqueSyclKernelBundle
    cdef struct DPCTLOpaqueSyclQueue
    cdef struct DPCTLOpaqueSyclUSM

    ctypedef DPCTLOpaqueSyclContext        *DPCTLSyclContextRef
    ctypedef DPCTLOpaqueSyclDevice         *DPCTLSyclDeviceRef
    ctypedef DPCTLOpaqueSyclDeviceSelector *DPCTLSyclDeviceSelectorRef
    ctypedef DPCTLOpaqueSyclEvent          *DPCTLSyclEventRef
    ctypedef DPCTLOpaqueSyclKernel         *DPCTLSyclKernelRef
    ctypedef DPCTLOpaqueSyclPlatform       *DPCTLSyclPlatformRef
    ctypedef DPCTLOpaqueSyclKernelBundle   *DPCTLSyclKernelBundleRef
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
    cdef size_t *DPCTLDevice_GetMaxWorkItemSizes1d(const DPCTLSyclDeviceRef DRef)
    cdef size_t *DPCTLDevice_GetMaxWorkItemSizes2d(const DPCTLSyclDeviceRef DRef)
    cdef size_t *DPCTLDevice_GetMaxWorkItemSizes3d(const DPCTLSyclDeviceRef DRef)
    cdef const char *DPCTLDevice_GetName(const DPCTLSyclDeviceRef DRef)
    cdef DPCTLSyclPlatformRef DPCTLDevice_GetPlatform(
        const DPCTLSyclDeviceRef DRef)
    cdef const char *DPCTLDevice_GetVendor(const DPCTLSyclDeviceRef DRef)
    cdef size_t DPCTLDevice_Hash(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsAccelerator(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsCPU(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_IsGPU(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_GetSubGroupIndependentForwardProgress(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthChar(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthShort(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthInt(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthLong(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthFloat(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthDouble(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetPreferredVectorWidthHalf(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetNativeVectorWidthChar(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetNativeVectorWidthShort(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetNativeVectorWidthInt(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetNativeVectorWidthLong(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetNativeVectorWidthFloat(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetNativeVectorWidthDouble(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetNativeVectorWidthHalf(const DPCTLSyclDeviceRef DRef)
    cdef bool DPCTLDevice_HasAspect(const DPCTLSyclDeviceRef, _aspect_type)
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
    cdef size_t DPCTLDevice_GetProfilingTimerResolution(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetGlobalMemCacheLineSize(const DPCTLSyclDeviceRef DRef)
    cdef uint64_t DPCTLDevice_GetGlobalMemCacheSize(const DPCTLSyclDeviceRef DRef)
    cdef _global_mem_cache_type DPCTLDevice_GetGlobalMemCacheType(
        const DPCTLSyclDeviceRef DRef)
    cdef size_t *DPCTLDevice_GetSubGroupSizes(const DPCTLSyclDeviceRef DRef,
        size_t *res_len)
    cdef uint32_t DPCTLDevice_GetPartitionMaxSubDevices(const DPCTLSyclDeviceRef DRef)
    cdef uint32_t DPCTLDevice_GetMaxClockFrequency(const DPCTLSyclDeviceRef DRef)
    cdef uint64_t DPCTLDevice_GetMaxMemAllocSize(const DPCTLSyclDeviceRef DRef)


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
    cdef size_t DPCTLKernel_GetNumArgs(const DPCTLSyclKernelRef KRef)
    cdef void DPCTLKernel_Delete(DPCTLSyclKernelRef KRef)
    cdef DPCTLSyclKernelRef DPCTLKernel_Copy(const DPCTLSyclKernelRef KRef)
    cdef size_t DPCTLKernel_GetWorkGroupSize(const DPCTLSyclKernelRef KRef)
    cdef size_t DPCTLKernel_GetPreferredWorkGroupSizeMultiple(const DPCTLSyclKernelRef KRef)
    cdef size_t DPCTLKernel_GetPrivateMemSize(const DPCTLSyclKernelRef KRef)
    cdef uint32_t DPCTLKernel_GetMaxNumSubGroups(const DPCTLSyclKernelRef KRef)
    cdef uint32_t DPCTLKernel_GetMaxSubGroupSize(const DPCTLSyclKernelRef KRef)
    cdef uint32_t DPCTLKernel_GetCompileNumSubGroups(const DPCTLSyclKernelRef KRef)
    cdef uint32_t DPCTLKernel_GetCompileSubGroupSize(const DPCTLSyclKernelRef KRef)


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
    cdef const char *DPCTLPlatformMgr_GetInfo(const DPCTLSyclPlatformRef, size_t)


cdef extern from "syclinterface/dpctl_sycl_platform_interface.h":
    cdef bool DPCTLPlatform_AreEq(const DPCTLSyclPlatformRef, const DPCTLSyclPlatformRef)
    cdef DPCTLSyclPlatformRef DPCTLPlatform_Copy(const DPCTLSyclPlatformRef)
    cdef DPCTLSyclPlatformRef DPCTLPlatform_Create()
    cdef DPCTLSyclPlatformRef DPCTLPlatform_CreateFromSelector(
        const DPCTLSyclDeviceSelectorRef)
    cdef void DPCTLPlatform_Delete(DPCTLSyclPlatformRef)
    cdef _backend_type DPCTLPlatform_GetBackend(const DPCTLSyclPlatformRef)
    cdef const char *DPCTLPlatform_GetName(const DPCTLSyclPlatformRef)
    cdef const char *DPCTLPlatform_GetVendor(const DPCTLSyclPlatformRef)
    cdef const char *DPCTLPlatform_GetVersion(const DPCTLSyclPlatformRef)
    cdef size_t DPCTLPlatform_Hash(const DPCTLSyclPlatformRef)
    cdef DPCTLPlatformVectorRef DPCTLPlatform_GetPlatforms()
    cdef DPCTLSyclContextRef DPCTLPlatform_GetDefaultContext(
        const DPCTLSyclPlatformRef)


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


cdef extern from "syclinterface/dpctl_sycl_kernel_bundle_interface.h":
    cdef DPCTLSyclKernelBundleRef DPCTLKernelBundle_CreateFromSpirv(
        const DPCTLSyclContextRef Ctx,
        const DPCTLSyclDeviceRef Dev,
        const void *IL,
        size_t Length,
        const char *CompileOpts)
    cdef DPCTLSyclKernelBundleRef DPCTLKernelBundle_CreateFromOCLSource(
        const DPCTLSyclContextRef Ctx,
        const DPCTLSyclDeviceRef Dev,
        const char *Source,
        const char *CompileOpts)
    cdef DPCTLSyclKernelRef DPCTLKernelBundle_GetKernel(
        DPCTLSyclKernelBundleRef KBRef,
        const char *KernelName)
    cdef bool DPCTLKernelBundle_HasKernel(DPCTLSyclKernelBundleRef KBRef,
                                     const char *KernelName)
    cdef void DPCTLKernelBundle_Delete(DPCTLSyclKernelBundleRef KBRef)
    cdef DPCTLSyclKernelBundleRef DPCTLKernelBundle_Copy(
        const DPCTLSyclKernelBundleRef KBRef)


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
    cdef DPCTLSyclEventRef DPCTLQueue_MemcpyWithEvents(
        const DPCTLSyclQueueRef Q,
        void *Dest,
        const void *Src,
        size_t Count,
        const DPCTLSyclEventRef *depEvents,
        size_t depEventsCount)
    cdef DPCTLSyclEventRef DPCTLQueue_Memset(
        const DPCTLSyclQueueRef Q,
        void *Dest,
        int Val,
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


cdef extern from "syclinterface/dpctl_sycl_usm_interface.h":
    cdef DPCTLSyclUSMRef DPCTLmalloc_shared(
        size_t size,
        DPCTLSyclQueueRef QRef) nogil
    cdef DPCTLSyclUSMRef DPCTLmalloc_host(
        size_t size,
        DPCTLSyclQueueRef QRef) nogil
    cdef DPCTLSyclUSMRef DPCTLmalloc_device(
        size_t size,
        DPCTLSyclQueueRef QRef) nogil
    cdef DPCTLSyclUSMRef DPCTLaligned_alloc_shared(
        size_t alignment,
        size_t size,
        DPCTLSyclQueueRef QRef) nogil
    cdef DPCTLSyclUSMRef DPCTLaligned_alloc_host(
        size_t alignment,
        size_t size,
        DPCTLSyclQueueRef QRef) nogil
    cdef DPCTLSyclUSMRef DPCTLaligned_alloc_device(
        size_t alignment,
        size_t size,
        DPCTLSyclQueueRef QRef) nogil
    cdef void DPCTLfree_with_queue(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclQueueRef QRef)
    cdef void DPCTLfree_with_context(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclContextRef CRef)
    cdef _usm_type DPCTLUSM_GetPointerType(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclContextRef CRef)
    cdef DPCTLSyclDeviceRef DPCTLUSM_GetPointerDevice(
        DPCTLSyclUSMRef MRef,
        DPCTLSyclContextRef CRef)
