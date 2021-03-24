//===-- dpctl_vector_templ.cpp - Wrapper functions for opaque vector types ===//
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
/// This file is meant to be directly included inside other source files to add
/// the wrapper functions for vector operations.
///
//===----------------------------------------------------------------------===//
#include "../helper/include/dpctl_vector_macros.h"
#include "Support/MemOwnershipAttrs.h"
#include <type_traits>

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(vector_class<SYCLREF(EL)>, VECTOR(EL))
}

/*!
 * @brief Creates a new std::vector of the opaque SYCL pointer types.
 *
 * @return   A new dynamically allocated std::vector of opaque pointer types.
 */
__dpctl_give VECTOR(EL) FN(EL, Create)()
{
    try {
        auto Vec = new vector_class<SYCLREF(EL)>();
        return wrap(Vec);
    } catch (std::bad_alloc const &ba) {
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
 * @brief Returns the number of elements in the vector.
 *
 */
void FN(EL, Resize)(__dpctl_keep VECTOR(EL) VRef, size_t resize)
{
    auto V = unwrap(VRef);
    if (V)
        V->resize(resize);
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
        try {
            auto ret = Vec->at(index);
            auto Ref = unwrap(ret);
            copy = wrap(new std::remove_pointer<decltype(Ref)>::type(*Ref));
        } catch (std::out_of_range const &oor) {
            std::cerr << oor.what() << '\n';
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
            return nullptr;
        }
    }
    return copy;
}

/*!
 * @brief Returns a copy of the opaque pointer at specified index, and throws
 * an out_of_range exception if the index is incorrect.
 *
 */
void FN(EL,
        SetAt)(__dpctl_keep VECTOR(EL) VRef, size_t index, SYCLREF(EL) element)
{
    auto Vec = unwrap(VRef);
    // SYCLREF(EL) copy = nullptr;
    if (Vec) {
        try {
            Vec->at(index) = element;
        } catch (std::out_of_range const &oor) {
            std::cerr << oor.what() << '\n';
        } catch (std::bad_alloc const &ba) {
            // \todo log error
            std::cerr << ba.what() << '\n';
            // return nullptr;
        }
    }
}
