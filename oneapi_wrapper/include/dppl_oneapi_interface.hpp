//===-- dppl_oneapi_interface.hpp - DPPL-SYCL interface ---*- C++ -*-------===//
//
//               Python Data Parallel Processing Library (PyDPPL)
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
/// This file contains the declaration of a C++ API to expose a lightweight SYCL
/// interface for the Python dppl package.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifdef _WIN32
#    ifdef DPPLOneapiInterface_EXPORTS
#        define DPPL_API __declspec(dllexport)
#    else
#        define DPPL_API __declspec(dllimport)
#    endif
#else
#    define DPPL_API
#endif
