##===------------- backend.pyx - DPPL interface ------*- Cython -*-------===##
##
##               Python Data Parallel Processing Library (PyDPPL)
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
## This file defines the Cython interface for the backend API of PyDPPL.
##
##===----------------------------------------------------------------------===##

from libcpp cimport bool


cdef extern from "dppl_utils.h":
    cdef void DPPLDeleteCString (const char *str)


cdef extern from "dppl_sycl_types.h":
    cdef struct DPPLOpaqueSyclContext
    cdef struct DPPLOpaqueSyclQueue
    cdef struct DPPLOpaqueSyclDevice
    cdef struct DPPLOpaqueSyclUSM

    ctypedef DPPLOpaqueSyclContext* DPPLSyclContextRef
    ctypedef DPPLOpaqueSyclQueue* DPPLSyclQueueRef
    ctypedef DPPLOpaqueSyclDevice* DPPLSyclDeviceRef
    ctypedef DPPLOpaqueSyclUSM* DPPLSyclUSMRef


cdef extern from "dppl_sycl_context_interface.h":
<<<<<<< HEAD
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
=======
    cdef void DPPLDeleteSyclContext (DPPLSyclContextRef CtxtRef) except +


cdef extern from "dppl_sycl_device_interface.h":
    cdef void DPPLDumpDeviceInfo (const DPPLSyclDeviceRef DRef) except +
    cdef void DPPLDeleteSyclDevice (DPPLSyclDeviceRef DRef) except +
    cdef void DPPLDumpDeviceInfo (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDeviceIsAccelerator (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDeviceIsCPU (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDeviceIsGPU (const DPPLSyclDeviceRef DRef) except +
    cdef bool DPPLDeviceIsHost (const DPPLSyclDeviceRef DRef) except +
    cdef const char* DPPLGetDeviceDriverInfo (const DPPLSyclDeviceRef DRef) \
    except +
    cdef const char* DPPLGetDeviceName (const DPPLSyclDeviceRef DRef) except +
    cdef const char* DPPLGetDeviceVendorName (const DPPLSyclDeviceRef DRef) \
    except +
    cdef bool DPPLGetDeviceHostUnifiedMemory (const DPPLSyclDeviceRef DRef) \
>>>>>>> master
    except +


cdef extern from "dppl_sycl_platform_interface.h":
    cdef size_t DPPLPlatform_GetNumPlatforms ()
    cdef void DPPLPlatform_DumpInfo ()


cdef extern from "dppl_sycl_queue_interface.h":
<<<<<<< HEAD
    cdef void DPPLQueue_Delete (DPPLSyclQueueRef QRef) except +
    cdef DPPLSyclContextRef DPPLQueue_GetContext (const DPPLSyclQueueRef Q) \
         except+
    cdef DPPLSyclDeviceRef DPPLQueue_GetDevice (const DPPLSyclQueueRef Q) \
=======
    cdef void DPPLDeleteSyclQueue (DPPLSyclQueueRef QRef) except +
    cdef DPPLSyclContextRef DPPLGetContextFromQueue (const DPPLSyclQueueRef Q) \
         except+
    cdef DPPLSyclDeviceRef DPPLGetDeviceFromQueue (const DPPLSyclQueueRef Q) \
>>>>>>> master
         except +


cdef extern from "dppl_sycl_queue_manager.h":
    cdef enum _device_type 'DPPLSyclDeviceType':
        _GPU 'DPPL_GPU'
        _CPU 'DPPL_CPU'

<<<<<<< HEAD
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
=======
    cdef DPPLSyclQueueRef DPPLGetCurrentQueue () except +
    cdef size_t DPPLGetNumCPUQueues () except +
    cdef size_t DPPLGetNumGPUQueues () except +
    cdef size_t DPPLGetNumActivatedQueues () except +
    cdef DPPLSyclQueueRef DPPLGetQueue (_device_type DTy,
                                        size_t device_num) except +
    cdef void DPPLPopSyclQueue () except +
    cdef DPPLSyclQueueRef DPPLPushSyclQueue (_device_type DTy,
                                             size_t device_num) except +
    cdef void DPPLSetAsDefaultQueue (_device_type DTy,
                                     size_t device_num) except +
>>>>>>> master


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
