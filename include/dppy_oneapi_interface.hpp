//===-- dppy_oneapi_interface.hpp - DPPY-SYCL interface ---*- C++ -*-------===//
//
//                     Data Parallel Python (DPPY)
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include <memory>
#include <CL/sycl.hpp>                /* SYCL headers */


////////////////////////////////////////////////////////////////////////////////
//////////////////////////////// DppyOneAPIContext /////////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*! \class DppyOneAPIContext
 *  \brief A convenience wrapper encapsulating a SYCL queue
 *
 */
class DppyOneAPIContext
{
    std::shared_ptr<cl::sycl::queue> queue_;

public:
    auto dump ();

    DppyOneAPIContext(const cl::sycl::device_selector & dev_sel
                          = cl::sycl::default_selector());
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////DppyOneAPIContextFactory ///////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*! \class DppyOneAPIContextFactory
 *  \brief A singleton class shared by all users DPPY
 *
 */
class DppyOneAPIContextFactory
{
    size_t num_platforms_;
    size_t num_cpus_;
    size_t num_gpus_;

public:
    auto getGPUContext     ()               const;
    auto getGPUContext     (size_t gpu_id)  const;
    auto getCPUContext     (size_t cpu_id)  const;
    auto getFPGAContext    (size_t fpga_id) const;
    auto dump              ()               const;

    DppyOneAPIContextFactory();
    ~DppyOneAPIContextFactory();
};

#endif /*--- DPPY_ONEAPI_INTERFACE_HPP_ ---*/
