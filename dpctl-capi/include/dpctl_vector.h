//===-- dpctl_list.h - Defines macros for opaque vector types      -*-C++-*- =//
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
/// A set of helper macros are defined here to create opaque lists (implemented
/// using std::vector) and helper functions of any DPCTL type.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"

DPCTL_C_EXTERN_C_BEGIN

#define DPCTL_DECLARE_VECTOR_TYPE(EL)                                          \
    typedef struct DPCTL##EL##Vector *DPCTL##EL##VectorRef;

#define DPCTL_DECLARE_VECTOR_FN(EL)                                            \
    DPCTL_API                                                                  \
    __dpctl_give DPCTL##EL##VectorRef DPCTL##EL##Vector_Create();              \
                                                                               \
    DPCTL_API                                                                  \
    void DPCTL##EL##Vector_Delete(__dpctl_take DPCTL##EL##VectorRef Ref);      \
                                                                               \
    DPCTL_API                                                                  \
    void DPCTL##EL##Vector_Clear(__dpctl_keep DPCTL##EL##VectorRef Ref);       \
                                                                               \
    DPCTL_API                                                                  \
    size_t DPCTL##EL##Vector_Size(__dpctl_keep DPCTL##EL##VectorRef Ref);      \
                                                                               \
    DPCTL_API                                                                  \
    size_t DPCTL##EL##Vector_GetAt(__dpctl_keep DPCTL##EL##VectorRef Ref,      \
                                   size_t index);

#define DPCTL_DECLARE_VECTOR(EL)                                               \
    DPCTL_DECLARE_VECTOR_TYPE(EL)                                              \
    DPCTL_DECLARE_VECTOR_FN(EL)

DPCTL_C_EXTERN_C_END
