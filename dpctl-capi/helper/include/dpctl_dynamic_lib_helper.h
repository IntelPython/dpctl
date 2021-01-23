//===--------------- dpctl_dynamic_lib_helper.h - dpctl-C_API     -*-C++-*-===//
//
//               Data Parallel Control Library (dpCtl)
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
/// Helper for dynamic libs management.
//===----------------------------------------------------------------------===//

#ifndef __DPCTL_DYNAMIC_LIB_HELPER_H__
#define __DPCTL_DYNAMIC_LIB_HELPER_H__

#if defined(__linux__) || defined(_WIN32) || defined(_WIN64)

    #ifdef __linux__

        #include <dlfcn.h>

    #elif defined(_WIN32) || defined(_WIN64)

        #define NOMINMAX
        #include <windows.h>

    #endif // __linux__

#include <cstdint>

namespace dpctl
{

class DynamicLibHelper final
{
public:
    DynamicLibHelper()                         = delete;
    DynamicLibHelper(const DynamicLibHelper &) = delete;
    DynamicLibHelper(const char * libName, int flag, int64_t & status)
    {

    #ifdef __linux__
        _handle = dlopen(libName, flag);
    #elif defined(_WIN32) || defined(_WIN64)
        _handle = LoadLibraryA(libName);
    #endif
        if (!_handle)
        {
            status = -1;
            return;
        }
    }

    ~DynamicLibHelper()
    {
    #ifdef __linux__
        dlclose(_handle);
    #elif defined(_WIN32) || defined(_WIN64)
        FreeLibrary((HMODULE)_handle);
    #endif
    };

    template <typename T>
    T getSymbol(const char * symName)
    {
    #ifdef __linux__
        void * sym   = dlsym(_handle, symName);
        char * error = dlerror();

        if (NULL != error)
        {
            return nullptr;
        }
    #elif defined(_WIN32) || defined(_WIN64)
        void * sym = (void *)GetProcAddress((HMODULE)_handle, symName);

        if (NULL == sym)
        {
            return nullptr;
        }
    #endif

        return (T)sym;
    }

private:
    void * _handle;
};

} // namespace dpctl

#endif // #if defined(__linux__) || defined(_WIN32) || defined(_WIN64)
#endif // __DPCTL_DYNAMIC_LIB_HELPER_H__
