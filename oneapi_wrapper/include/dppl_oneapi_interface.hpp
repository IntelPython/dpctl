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

#include <cstdlib>
#include <deque>

#ifdef _WIN32
#    ifdef DPPLOneapiInterface_EXPORTS
#        define DPPL_API __declspec(dllexport)
#    else
#        define DPPL_API __declspec(dllimport)
#    endif
#else
#    define DPPL_API
#endif

namespace dppl
{

/*!
 * Redefinition of Sycl's device_type so that we do not have to include
 * sycl.hpp here, and in the Python bindings.
 */
enum class sycl_device_type : unsigned int
{
    cpu,
    gpu,
    accelerator,
    custom,
    automatic,
    host,
    all
};


////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DpplOneAPIRuntime /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*! \class DpplOneAPIRuntime
 *  \brief A runtime and context factory class
 *
 */
class DpplOneAPIRuntime
{
public:
    DPPL_API
    int64_t getNumPlatforms (size_t *platforms) const;
    DPPL_API
    int64_t getCurrentQueue (void **Q) const;
    DPPL_API
    int64_t getQueue (void **Q,
                      dppl::sycl_device_type DeviceTy,
                      size_t DNum = 0) const;
    DPPL_API
    int64_t resetGlobalQueue (dppl::sycl_device_type DeviceTy,
                              size_t DNum = 0);
    /*!
     * Push a new sycl queue to the top of the activate_queues deque. The
     * newly activated queue is returned to caller inside the Q object.
     */
    DPPL_API
    int64_t activateQueue (void **Q,
                           dppl::sycl_device_type DeviceTy,
                           size_t DNum);
    DPPL_API
    int64_t deactivateCurrentQueue ();
    DPPL_API
    int64_t dump () const;
    DPPL_API
    int64_t dump_queue (const void *Q) const;
};


DPPL_API
int64_t deleteQueue (void *Q);

} /* end of namespace dppl */
