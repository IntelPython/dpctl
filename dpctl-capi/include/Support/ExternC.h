//===----------- ExternC.h - Defines a extern C helper macro     -*-C++-*- ===//
//
//                      Data Parallel Control (dpCtl)
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
/// This file defines an extern "C" wrapper.
///
//===----------------------------------------------------------------------===//

#pragma once

#ifdef __cplusplus
#define DPCTL_C_EXTERN_C_BEGIN  extern "C" {
#define DPCTL_C_EXTERN_C_END    }
#else
#define DPCTL_C_EXTERN_C_BEGIN
#define DPCTL_C_EXTERN_C_END
#endif
