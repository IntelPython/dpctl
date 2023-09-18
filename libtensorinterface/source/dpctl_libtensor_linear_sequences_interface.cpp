#include <vector>
#include <utility>
#include <CL/sycl.hpp>
#include <pybind11/pybind11.h>

#include "dpctl_libtensor_linear_sequences_interface.h"
#include "../../dpctl/tensor/libtensor/source/linear_sequences.hpp"

using py = pybind11;

DPCTLSyclEventPair 
DPCTL_usm_ndarray_linear_sequence_affine_int(uint64_t start, 
                                            uint64_t end, 
                                            char *data,
                                            uint64_t include_endpoint, 
                                            const DPCTLSyclQueueRef QRef,
                                            void *events)
{
    /**
    TODO:
        Call this function:
        std::pair<sycl::event, sycl::event> usm_ndarray_linear_sequence_affine(
                                py::object start,
                                py::object end,
                                dpctl::tensor::usm_ndarray dst,
                                bool include_endpoint,
                                sycl::queue exec_q,
                                const std::vector<sycl::event> &depends = {});

        1. convert start and end into py::object
        2. convert char *data into dpctl::tensor::usm_ndarray
        3. convert QRef into sycl::queue
        4. convert *events into std::vector<sycl::event>&
    */
    std::pair<sycl::event, sycl::event> epair;
    DPCTLSyclEventPair epair_c;
    epair_c.first = epair.first;
    epair_c.second = epair.second;
    return epair_c;
}
