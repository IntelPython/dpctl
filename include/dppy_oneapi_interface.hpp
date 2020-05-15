//===-- dppy_oneapi_interface.hpp - DPPY-SYCL interface ---*- C++ -*-------===//
//
//                     Data Parallel Python (DPPY)
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
/// interface for the Python dppy package.
///
//===----------------------------------------------------------------------===//

#ifndef DPPY_ONEAPI_INTERFACE_HPP_
#define DPPY_ONEAPI_INTERFACE_HPP_

#include <deque>
#include <memory>
#include <CL/sycl.hpp>                /* SYCL headers */

namespace dppy_rt
{
////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIContext /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

enum class ErrorCode
{
    DPPY_SUCCESS, DPPY_FAILURE
};

/*! \class DppyOneAPIContext
 *  \brief A convenience wrapper encapsulating a SYCL queue
 *
 */
class DppyOneAPIContext
{
    cl::sycl::queue queue_;

public:

    auto getSyclQueue     () const;
    auto getSyclContext   () const;
    auto getSyclDevice    () const;
    auto getOpenCLQueue   () const;
    auto getOpenCLContext () const;
    auto getOpenCLDevice  () const;

    auto dump ();

    DppyOneAPIContext(const cl::sycl::device_selector & dev_sel
                          = cl::sycl::default_selector());
    DppyOneAPIContext(const cl::sycl::device & dev);
};


////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIRuntime /////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*! \class DppyOneAPIRuntime
 *  \brief A runtime and context factory class
 *
 */
class DppyOneAPIRuntime
{
    size_t                                   num_platforms_;
    cl::sycl::vector_class<cl::sycl::device> cpu_contexts_;
    cl::sycl::vector_class<cl::sycl::device> gpu_contexts_;
    std::deque<DppyOneAPIContext>            available_contexts_;

public:
    ErrorCode getDefaultContext (DppyOneAPIContext * ctx) const;
    ErrorCode getCurrentContext (DppyOneAPIContext * ctx) const;
    ErrorCode setCurrentContext (cl::sycl::info::device_type ty,
                                 size_t device_num);
    ErrorCode dump              ()                        const;

    DppyOneAPIRuntime();
    ~DppyOneAPIRuntime();
};

} /* end of namespace dppy_rt */

#endif /*--- DPPY_ONEAPI_INTERFACE_HPP_ ---*/
