//===- MemOwnershipAttrs.h - Defines memory ownership attributes -*-C++-*- ===//
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
/// This file defines a group of macros that serve as attributes indicating the
/// type of ownership of a pointer. The macros are modeled after similar
/// attributes defined in Integer Set Library (isl) and serve the purpose of
/// helping a programmer understand the semantics of a dpctl function.
///
//===----------------------------------------------------------------------===//

#pragma once

/**
 * @defgroup MemorySemantics Self-documenting memory ownership semantics tokens
 */

/**
 * @def __dpctl_give
 * @brief The __dpctl_give attribute indicates that a new object is returned and
 * the caller now owns the object.
 *
 * The __dpctl_give attribute informs a user that the function is allocating a
 * new object and returning it to the user. The user now owns the object and to
 * free the object, he/she should make sure to use it exactly once as a value
 * for a __dpctl_take argument. However, the user is free to use the object as
 * he/she likes as a value to __dpctl_keep arguments.
 *
 * @ingroup MemorySemantics
 */
#ifndef __dpctl_give
#define __dpctl_give
#endif

/*!
 * @def __dpctl_take
 * @brief The __dpctl_take attribute indicates that the function "takes" over
 * the ownership of the object and the user must not use the object as an
 * argument to another function.
 *
 * The __dpctl_take attribute means that the function destroys it before the
 * function returns, and the caller must not use the object again in any other
 * function. If the pointer annotated with __dpctl_take is NULL then it is
 * treated as an error, since it may prevent the normal behavior of the
 * function.
 *
 * @ingroup MemorySemantics
 */
#ifndef __dpctl_take
#define __dpctl_take
#endif

/*!
 * @def __dpctl_keep
 * @brief The __dpctl_keep attribute indicates that the function only uses the
 * object and does not destroy it before returning.
 *
 * @ingroup MemorySemantics
 */
#ifndef __dpctl_keep
#define __dpctl_keep
#endif

/*!
 * @def __dpctl_null
 * @brief The __dpctl_null attribute indicates that a NULL value is returned.
 * @ingroup MemorySemantics
 */
#ifndef __dpctl_null
#define __dpctl_null
#endif
