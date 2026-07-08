//===- dpctl_sycl_ipc_memory_interface.cpp - C API for SYCL IPC memory ----===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2026 Intel Corporation
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
/// This file implements the functions declared in
/// dpctl_sycl_ipc_memory_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_ipc_memory_interface.h"
#include "Config/dpctl_config.h"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_type_casters.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

// Support both the new namespace (ipc::memory, oneAPI >= 2026.1) and the
// deprecated namespace (ipc_memory, oneAPI 2026.0).
#if __has_include(<sycl/ext/oneapi/experimental/detail/ipc_common.hpp>)
// New layout: ipc::memory namespace with separate ipc::handle/handle_data_t
namespace ipc = sycl::ext::oneapi::experimental::ipc::memory;
using ipc_handle_data_t = sycl::ext::oneapi::experimental::ipc::handle_data_t;
#else
// Old layout: everything in ipc_memory namespace
namespace ipc = sycl::ext::oneapi::experimental::ipc_memory;
using ipc_handle_data_t = ipc::handle_data_t;
#endif

namespace
{
static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VERSION_REQUIRED,
              "The compiler does not meet minimum version requirement");

using namespace dpctl::syclinterface;
} // end of anonymous namespace

int DPCTLIPCMem_GetHandle(__dpctl_keep DPCTLSyclUSMRef Ptr,
                          __dpctl_keep const DPCTLSyclContextRef CRef,
                          char **DataOut,
                          size_t *SizeOut)
{
    if (!Ptr) {
        error_handler("Input Ptr is nullptr.", __FILE__, __func__, __LINE__);
        return 1;
    }
    if (!CRef) {
        error_handler("Input CRef is nullptr.", __FILE__, __func__, __LINE__);
        return 1;
    }
    if (!DataOut || !SizeOut) {
        error_handler("Output pointers are nullptr.", __FILE__, __func__,
                      __LINE__);
        return 1;
    }

    try {
        auto *RawPtr = unwrap<void>(Ptr);
        auto *Ctx = unwrap<context>(CRef);

        // Obtain the IPC handle from the SYCL runtime.
        auto Handle = ipc::get(RawPtr, *Ctx);

        // Copy handle data into a malloc'd buffer for the caller.
        auto HandleData = Handle.data(); // std::vector<std::byte>

        size_t Size = HandleData.size();
        char *Buf = static_cast<char *>(std::malloc(Size));
        if (!Buf) {
            error_handler("Failed to allocate handle data buffer.", __FILE__,
                          __func__, __LINE__);
            return 1;
        }
        std::memcpy(Buf, HandleData.data(), Size);

        *DataOut = Buf;
        *SizeOut = Size;
        return 0;
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return 1;
    }
}

__dpctl_give DPCTLSyclUSMRef
DPCTLIPCMem_OpenHandle(const char *HandleData,
                       size_t HandleDataSize,
                       __dpctl_keep const DPCTLSyclContextRef CRef,
                       __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    if (!HandleData) {
        error_handler("Input HandleData is nullptr.", __FILE__, __func__,
                      __LINE__);
        return nullptr;
    }
    if (!CRef) {
        error_handler("Input CRef is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }
    if (!DRef) {
        error_handler("Input DRef is nullptr.", __FILE__, __func__, __LINE__);
        return nullptr;
    }

    try {
        auto *Ctx = unwrap<context>(CRef);
        auto *Dev = unwrap<device>(DRef);

        // Rebuild handle_data_t (vector<byte>) from the raw byte buffer.
        ipc_handle_data_t HData(
            reinterpret_cast<const std::byte *>(HandleData),
            reinterpret_cast<const std::byte *>(HandleData) + HandleDataSize);

        void *MappedPtr = ipc::open(HData, *Ctx, *Dev);
        return wrap<void>(MappedPtr);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

void DPCTLIPCMem_CloseHandle(__dpctl_keep DPCTLSyclUSMRef MappedPtr,
                             __dpctl_keep const DPCTLSyclContextRef CRef)
{
    if (!MappedPtr) {
        error_handler("Input MappedPtr is nullptr.", __FILE__, __func__,
                      __LINE__);
        return;
    }
    if (!CRef) {
        error_handler("Input CRef is nullptr.", __FILE__, __func__, __LINE__);
        return;
    }

    try {
        auto *RawPtr = unwrap<void>(MappedPtr);
        auto *Ctx = unwrap<context>(CRef);
        ipc::close(RawPtr, *Ctx);
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }
}

void DPCTLIPCMem_FreeHandleData(char *Data) { std::free(Data); }
