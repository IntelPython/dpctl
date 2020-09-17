//===----- error_check_macros.h - DPPL-OpenCL interface -------*- C -*-----===//
//
//               Python Data Parallel Processing Python (PyDPPL)
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
/// This file contains a set of macros to check for different OpenCL error
/// codes.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <stdio.h>

// TODO : Add branches to check for OpenCL error codes and print relevant error
//        messages. Then there would be no need to pass in the message string

// FIXME : The error check macro needs to be improved. Currently, we encounter
// an error and goto the error label. Directly going to the error label can lead
// to us not releasing resources prior to returning from the function. To work
// around this situation, add a stack to store all the objects that should be
// released prior to returning. The stack gets populated as a function executes
// and on encountering an error, all objects on the stack get properly released
// prior to returning. (Look at enqueue_dp_kernel_from_source for a
// ghastly example where we really need proper resource management.)

// FIXME : memory allocated in a function should be released in the error
// section

#define CHECK_OPEN_CL_ERROR(x, M) do {                                         \
    int retval = (x);                                                          \
    switch(retval) {                                                           \
    case 0:                                                                    \
        break;                                                                 \
    case -36:                                                                  \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, "[CL_INVALID_COMMAND_QUEUE]command_queue is not a "    \
                        "valid command-queue.",                                \
                __LINE__, __FILE__);                                           \
        goto error;                                                            \
    case -38:                                                                  \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n"    \
                        "%s\n",                                                \
                retval, "[CL_INVALID_MEM_OBJECT] memory object is not a "      \
                        "valid OpenCL memory object.",                         \
                __LINE__, __FILE__,M);                                         \
        goto error;                                                            \
    case -45:                                                                  \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, "[CL_INVALID_PROGRAM_EXECUTABLE] no successfully "     \
                        "built program executable available for device "       \
                        "associated with command_queue.",                      \
                __LINE__, __FILE__);                                           \
        goto error;                                                            \
    case -54:                                                                  \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, "[CL_INVALID_WORK_GROUP_SIZE]",                        \
                __LINE__, __FILE__);                                           \
        goto error;                                                            \
    default:                                                                   \
        fprintf(stderr, "Open CL Runtime Error: %d (%s) on Line %d in %s\n",   \
                retval, M, __LINE__, __FILE__);                                \
        goto error;                                                            \
    }                                                                          \
} while(0)


#define CHECK_MALLOC_ERROR(type, x) do {                                       \
    type * ptr = (type*)(x);                                                   \
    if(ptr == NULL) {                                                          \
        fprintf(stderr, "Malloc Error for type %s on Line %d in %s",           \
                #type, __LINE__, __FILE__);                                    \
        perror(" ");                                                           \
        free(ptr);                                                             \
        ptr = NULL;                                                            \
        goto malloc_error;                                                     \
    }                                                                          \
} while(0)


#define CHECK_DPGLUE_ERROR(x, M) do {                                          \
    int retval = (x);                                                          \
    switch(retval) {                                                           \
    case 0:                                                                    \
        break;                                                                 \
    case -1:                                                                   \
        fprintf(stderr, "DP_Glue Error: %d (%s) on Line %d in %s\n",           \
                retval, M, __LINE__, __FILE__);                                \
        goto error;                                                            \
    default:                                                                   \
        fprintf(stderr, "DP_Glue Error: %d (%s) on Line %d in %s\n",           \
                retval, M, __LINE__, __FILE__);                                \
        goto error;                                                            \
    }                                                                          \
} while(0)
