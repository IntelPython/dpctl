//===-- dpctl_vector_templ.cpp - Wrapper functions for opaque vector types ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
/// This file is meant to be directly included inside other source files to add
/// the wrapper functions for vector operations.
///
//===----------------------------------------------------------------------===//
#include "Support/MemOwnershipAttrs.h"
#include "dpctl_error_handlers.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_vector_macros.h"
#include <type_traits>
#include <vector>

/*!
 * @brief Creates a new std::vector of the opaque SYCL pointer types.
 *
 * @return   A new dynamically allocated std::vector of opaque pointer types.
 */
__dpctl_give VECTOR(EL) FN(EL, Create)()
{
    using vecTy = std::vector<SYCLREF(EL)>;
    vecTy *Vec = nullptr;
    try {
        Vec = new std::vector<SYCLREF(EL)>();
        return wrap<vecTy>(Vec);
    } catch (std::exception const &e) {
        delete Vec;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

/*!
 * @brief Creates a new std::vector of the opaque SYCL pointer types from given
 * C array with deep copy.
 *
 * @return   A new dynamically allocated std::vector of opaque pointer types.
 */
__dpctl_give VECTOR(EL)
    FN(EL, CreateFromArray)(size_t n, __dpctl_keep SYCLREF(EL) * elems)
{
    using vecTy = std::vector<SYCLREF(EL)>;
    vecTy *Vec = nullptr;
    try {
        Vec = new vecTy();
        for (size_t i = 0; i < n; ++i) {
            auto Ref = unwrap(elems[i]);
            Vec->emplace_back(
                wrap(new std::remove_pointer<decltype(Ref)>::type(*Ref)));
        }
        return wrap<vecTy>(Vec);
    } catch (std::exception const &e) {
        delete Vec;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}

/*!
 * @brief Frees all the elements of the passed in std::vector and then frees the
 * std::vector pointer.
 *
 */
void FN(EL, Delete)(__dpctl_take VECTOR(EL) VRef)
{
    auto Vec = unwrap(VRef);
    if (Vec) {
        for (auto i = 0ul; i < Vec->size(); ++i) {
            auto D = unwrap((*Vec)[i]);
            delete D;
        }
    }
    delete Vec;
}

/*!
 * @brief Frees all the elements of the vector and then calls clear().
 *
 */
void FN(EL, Clear)(__dpctl_keep VECTOR(EL) VRef)
{
    auto Vec = unwrap(VRef);
    if (Vec) {
        for (auto i = 0ul; i < Vec->size(); ++i) {
            auto D = unwrap((*Vec)[i]);
            delete D;
        }
        Vec->clear();
    }
}

/*!
 * @brief Returns the number of elements in the vector.
 *
 */
size_t FN(EL, Size)(__dpctl_keep VECTOR(EL) VRef)
{
    auto V = unwrap(VRef);
    if (V)
        return V->size();
    else
        return 0;
}

/*!
 * @brief Returns a copy of the opaque pointer at specified index, and throws
 * an out_of_range exception if the index is incorrect.
 *
 */
SYCLREF(EL) FN(EL, GetAt)(__dpctl_keep VECTOR(EL) VRef, size_t index)
{
    auto Vec = unwrap(VRef);
    SYCLREF(EL) copy = nullptr;
    if (Vec) {
        SYCLREF(EL) ret;
        try {
            ret = Vec->at(index);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
        auto Ref = unwrap(ret);
        std::remove_pointer<decltype(Ref)>::type *elPtr = nullptr;
        try {
            elPtr = new std::remove_pointer<decltype(Ref)>::type(*Ref);
            copy = wrap(elPtr);
        } catch (std::exception const &e) {
            delete elPtr;
            error_handler(e, __FILE__, __func__, __LINE__);
            return nullptr;
        }
    }
    return copy;
}
