#pragma once

#include "ExternC.h"
#include "dpctl_sycl_types.h"

DPCTL_C_EXTERN_C_BEGIN

typedef struct DPCTLSyclEventPair {
    DPCTLSyclEventRef first;
    DPCTLSyclEventRef second;
} DPCTLSyclEventPair;

DPCTLSyclEventPair 
DPCTL_usm_ndarray_linear_sequence_affine_int(uint64_t start, 
                                            uint64_t end, 
                                            char *data;
                                            uint64_t include_endpoint, 
                                            const DPCTLSyclQueueRef QRef,
                                            void *events);

DPCTL_C_EXTERN_C_END