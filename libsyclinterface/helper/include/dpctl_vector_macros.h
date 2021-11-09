//=== dpctl_vector_macros.h - Macros to help build function sig.    -*-C++-*- //
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
/// A set of macros used inside dpctl_vector_templ.cpp. The macros help build
/// the function signatures for the functions defined in dpctl_vector_templ.cpp.
///
//===----------------------------------------------------------------------===//

#pragma once

#define xFN(TYPE, NAME) DPCTL##TYPE##Vector##_##NAME
#define FN(TYPE, NAME) xFN(TYPE, NAME)
#define xVECTOR(EL) DPCTL##EL##VectorRef
#define VECTOR(EL) xVECTOR(EL)
#define xSYCLREF(EL) DPCTLSycl##EL##Ref
#define SYCLREF(EL) xSYCLREF(EL)
