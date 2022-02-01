//===--- dpctl_string_utils.hpp - C++ to C string converted     -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
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
/// Helper function to convert a C++ string to a C string.
//===----------------------------------------------------------------------===//
#include "dpctl_error_handlers.h"
#include <cstring>
#include <iostream>
#include <string>

#pragma once

namespace dpctl
{
namespace helper
{
/*!
 * @brief Convert a C++ std::string to a const char* and return the string to
 * caller.
 *
 * @param    str            A C++ string that has to be converted to a C string.
 * @return A const char* string representation of the C++ string.
 */
static inline __dpctl_give const char *
cstring_from_string(const std::string &str)
{
    char *cstr = nullptr;
    try {
        auto cstr_len = str.length() + 1;
        cstr = new char[cstr_len];
#ifdef _WIN32
        strncpy_s(cstr, cstr_len, str.c_str(), cstr_len);
#else
        std::strncpy(cstr, str.c_str(), cstr_len);
#endif
        // Added to resolve CheckMarx's false positive.
        // NB: This is redundant because str.c_str() is guaranteed
        // to be null-terminated and the copy function is asked to
        // copy enough characters to include that null-character.
        cstr[cstr_len - 1] = '\0';
    } catch (std::exception const &e) {
        error_handler(e, __FILE__, __func__, __LINE__);
    }

    return cstr;
}
} // namespace helper
} // namespace dpctl
