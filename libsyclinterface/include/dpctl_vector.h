//===-- dpctl_vector.h - Defines macros for opaque vector types    -*-C++-*- =//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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
/// using std::vector) and helper functions of any dpctl type.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "Support/DllExport.h"
#include "Support/ExternC.h"
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_data_types.h"

DPCTL_C_EXTERN_C_BEGIN

/*!
 * @brief Declare an opaque pointer type for a std::vector of opaque pointers
 * wrapping SYCL data types.
 */
#define DPCTL_DECLARE_VECTOR_TYPE(EL)                                          \
    typedef struct DPCTL##EL##Vector *DPCTL##EL##VectorRef;

#define DPCTL_DECLARE_VECTOR_FN(EL)                                            \
    /*!                                                                        \
       @brief Create an opaque pointer to a std::vector of opaque pointers     \
              wrapping the SYCL data type.                                     \
       @return Returns a new opaque pointer to a vector.                       \
     */                                                                        \
    DPCTL_API                                                                  \
    __dpctl_give DPCTL##EL##VectorRef DPCTL##EL##Vector_Create(void);          \
    /*!                                                                        \
       @brief Create an opaque pointer to a std::vector created from the       \
       input raw array. The elements of the input array are deep copied before \
       inserting the copies into the vector.                                   \
       @param len    Number of elements in the input array.                    \
       @param elems  A C array whose elements will be copied into the returned \
                     vector.                                                   \
       @return Returns a new opaque pointer to a vector.                       \
     */                                                                        \
    DPCTL_API                                                                  \
    __dpctl_give DPCTL##EL##VectorRef DPCTL##EL##Vector_CreateFromArray(       \
        size_t len, __dpctl_keep DPCTLSycl##EL##Ref *elems);                   \
                                                                               \
    /*!                                                                        \
       @brief Delete all elements in the vector and then delete the vector.    \
       @param VRef Opaque pointer to a vector to be deleted.                   \
     */                                                                        \
    DPCTL_API                                                                  \
    void DPCTL##EL##Vector_Delete(__dpctl_take DPCTL##EL##VectorRef VRef);     \
    /*!                                                                        \
       @brief Delete all the elements of the std::vector                       \
       @param VRef Opaque pointer to a vector.                                 \
     */                                                                        \
    DPCTL_API                                                                  \
    void DPCTL##EL##Vector_Clear(__dpctl_keep DPCTL##EL##VectorRef VRef);      \
    /*!                                                                        \
       @brief Returns the number of elements in the vector.                    \
       @param VRef Opaque pointer to a vector.                                 \
       @return The current size of the vector.                                 \
     */                                                                        \
    DPCTL_API                                                                  \
    size_t DPCTL##EL##Vector_Size(__dpctl_keep DPCTL##EL##VectorRef VRef);     \
    /*!                                                                        \
       @brief Returns the element at the specified index.                      \
       @param VRef Opaque pointer to a vector.                                 \
       @param index The index position of the element to be returned.          \
       @return The element at the specified position, if the index position is \
               out of bounds then a nullptr is returned.                       \
     */                                                                        \
    DPCTL_API                                                                  \
    __dpctl_give DPCTLSycl##EL##Ref DPCTL##EL##Vector_GetAt(                   \
        __dpctl_keep DPCTL##EL##VectorRef VRef, size_t index);

#define DPCTL_DECLARE_VECTOR(EL)                                               \
    DPCTL_DECLARE_VECTOR_TYPE(EL)                                              \
    DPCTL_DECLARE_VECTOR_FN(EL)

DPCTL_C_EXTERN_C_END
