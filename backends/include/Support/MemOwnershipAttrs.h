//===----- dppl_mem_ownership_attrs.h - DPPL-SYCL interface --*-- C++ --*--===//
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
/// This file defines a group of macros that serve as attributes indicating the
/// type of ownership of a pointer. The macros are modeled after similar
/// attributes defines in Integer Set Library (isl) and serve the purpose of
/// helping a programmer understand the semantics of a DPPL function.
///
//===----------------------------------------------------------------------===//
#pragma once

/**
 * @defgroup MEM_MGMT_ATTR_MACROS Memory management attributes.
 *
 * @{
 */

/*!
 * @def __dppl_give
 * @brief The __dppl_give attribute indicates that a new object is returned and
 * the caller now owns the object.
 *
 * The __dppl_give attribute informs a user that the function is allocating a
 * new object and returning it to the user. The user now owns the object and to
 * free the object, he/she should make sure to use it exactly once as a value
 * for a __dppl_take argument. However, the user is free to use the object as
 * he/she likes as a value to __dppl_keep arguments.
 *
 */
#ifndef __dppl_give
#define __dppl_give
#endif
/*!
 * @def __dppl_take
 * @brief The __dppl_take attribute indicates that the function "takes" over the
 * ownership of the object and the user must not use the object as an argument
 * to another function.
 *
 * The __dppl_take attribute mens that the function destroys it before the
 * function returns, and the caller must not use the object again in any other
 * function. If the pointer annotated with __dppl_take is NULL then it is
 * treated as an error, since it may prevent the normal behavior of the
 * function.
 *
 */
#ifndef __dppl_take
#define __dppl_take
#endif
/*!
 * @def __dppl_keep
 * @brief The __dppl_keep attribute indicates that the function only uses the
 * object and does not destroy it before returning.
 *
 */
#ifndef __dppl_keep
#define __dppl_keep
#endif
/*!
 * @def __dppl_null
 * @brief The __dppl_null attribute indicates that a NULL value is returned.
 *
 */
#ifndef __dppl_null
#define __dppl_null
#endif

/** @} */
