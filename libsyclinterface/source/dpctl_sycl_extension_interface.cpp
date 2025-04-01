//===---- dpctl_sycl_extension_interface.cpp - Implements C API for SYCL ext =//
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
/// This file implements the data types and functions declared in
/// dpctl_sycl_extension_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_extension_interface.h"

#include "dpctl_error_handlers.h"
#include "dpctl_sycl_type_casters.hpp"

#include <sycl/sycl.hpp>

using namespace dpctl::syclinterface;

DPCTL_API
__dpctl_give DPCTLSyclWorkGroupMemoryRef
DPCTLWorkGroupMemory_Create(size_t nbytes)
{
    DPCTLSyclWorkGroupMemoryRef wgm = nullptr;
    try {
        auto WorkGroupMem = new RawWorkGroupMemory{nbytes};
        wgm = wrap<RawWorkGroupMemory>(WorkGroupMem);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
    return wgm;
}

DPCTL_API
void DPCTLWorkGroupMemory_Delete(__dpctl_take DPCTLSyclWorkGroupMemoryRef Ref)
{
    delete unwrap<RawWorkGroupMemory>(Ref);
}

DPCTL_API
bool DPCTLWorkGroupMemory_Available()
{
#ifdef SYCL_EXT_ONEAPI_WORK_GROUP_MEMORY
    return true;
#else
    return false;
#endif
}

DPCTL_API
__dpctl_give DPCTLSyclRawKernelArgRef DPCTLRawKernelArg_Create(void *bytes,
                                                               size_t count)
{
    DPCTLSyclRawKernelArgRef rka = nullptr;
    try {
        auto RawKernelArg = std::unique_ptr<RawKernelArgData>(
            new RawKernelArgData(bytes, count));
        rka = wrap<RawKernelArgData>(RawKernelArg.get());
        RawKernelArg.release();
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
    return rka;
}

DPCTL_API
void DPCTLRawKernelArg_Delete(__dpctl_take DPCTLSyclRawKernelArgRef Ref)
{
    delete unwrap<RawKernelArgData>(Ref);
}

DPCTL_API
bool DPCTLRawKernelArg_Available()
{
#ifdef SYCL_EXT_ONEAPI_RAW_KERNEL_ARG
    return true;
#else
    return false;
#endif
}
