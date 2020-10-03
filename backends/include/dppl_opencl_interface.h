//===-- dppl_opencl_interface.h - DPPL-OpenCL interface -------*- C -*-----===//
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
/// This file contains the declaration of a C API to expose a lightweight OpenCL
/// interface for the Python dpctl package.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <stdbool.h>
#include <stdlib.h>

#ifdef _WIN32
#ifdef DPPLOpenCLInterface_EXPORTS
#define DPPL_API __declspec(dllexport)
#else
#define DPPL_API __declspec(dllimport)
#endif
#else
#define DPPL_API
#endif

enum DP_GLUE_ERROR_CODES { DP_GLUE_SUCCESS = 0, DP_GLUE_FAILURE = -1 };

/*!
 *
 */
struct dp_env {
  unsigned id_;
  // TODO : Add members to store more device related information such as name
  void *context;
  void *device;
  void *queue;
  unsigned int max_work_item_dims;
  size_t max_work_group_size;
  int support_int64_atomics;
  int support_float64_atomics;
  int (*dump_fn)(void *);
};

typedef struct dp_env *env_t;

struct dp_buffer {
  unsigned id_;
  // This may, for example, be a cl_mem pointer
  void *buffer_ptr;
  // Stores the size of the buffer_ptr (e.g sizeof(cl_mem))
  size_t sizeof_buffer_ptr;
};

typedef struct dp_buffer *buffer_t;

struct dp_kernel {
  unsigned id_;
  void *kernel;
  int (*dump_fn)(void *);
};

typedef struct dp_kernel *kernel_t;

struct dp_program {
  unsigned id_;
  void *program;
};

typedef struct dp_program *program_t;

struct dp_kernel_arg {
  unsigned id_;
  const void *arg_value;
  size_t arg_size;
};

typedef struct dp_kernel_arg *kernel_arg_t;

/*! @struct dp_runtime_t
 *  @brief Stores an array of the available OpenCL or Level-0 platform/drivers.
 *
 *  @var dp_runtime_t::num_platforms
 *  Depicts the number of platforms/drivers available on this system
 *
 *  @var dp_runtime_t::platforms_ids
 *  An array of OpenCL platforms.
 *
 */
struct dp_runtime {
  unsigned id_;
  unsigned num_platforms;
  void *platform_ids;
  bool has_cpu;
  bool has_gpu;
  env_t first_cpu_env;
  env_t first_gpu_env;
  env_t curr_env;
  int (*dump_fn)(void *);
};

typedef struct dp_runtime *runtime_t;

DPPL_API
int set_curr_env(runtime_t rt, env_t env);

/*!
 * @brief Initializes a new dp_runtime_t object
 *
 * @param[in/out] rt - An uninitialized runtime_t pointer that is initialized
 *                     by the function.
 *
 * @return An error code indicating if the runtime_t object was successfully
 *         initialized.
 */
DPPL_API
int create_dp_runtime(runtime_t *rt);

/*!
 * @brief Free the runtime and all its resources.
 *
 * @param[in] rt - Pointer to the numba_one_api_runtime_t object to be freed
 *
 * @return An error code indicating if resource freeing was successful.
 */
DPPL_API
int destroy_dp_runtime(runtime_t *rt);

/*!
 *
 */
DPPL_API
int create_dp_rw_mem_buffer(env_t env_t_ptr, size_t buffsize, buffer_t *buff);

DPPL_API
int destroy_dp_rw_mem_buffer(buffer_t *buff);

/*!
 *
 */
DPPL_API
int write_dp_mem_buffer_to_device(env_t env_t_ptr, buffer_t buff,
                                  bool blocking_copy, size_t offset,
                                  size_t buffersize, const void *data_ptr);

/*!
 *
 */
DPPL_API
int read_dp_mem_buffer_from_device(env_t env_t_ptr, buffer_t buff,
                                   bool blocking_copy, size_t offset,
                                   size_t buffersize, void *data_ptr);

/*!
 *
 */
DPPL_API
int create_dp_program_from_spirv(env_t env_t_ptr, const void *il, size_t length,
                                 program_t *program_t_ptr);

/*!
 *
 */
DPPL_API
int create_dp_program_from_source(env_t env_t_ptr, unsigned int count,
                                  const char **strings, const size_t *lengths,
                                  program_t *program_t_ptr);

/*!
 *
 */
DPPL_API
int destroy_dp_program(program_t *program_t_ptr);

DPPL_API
int build_dp_program(env_t env_t_ptr, program_t program_t_ptr);

/*!
 *
 */
DPPL_API
int create_dp_kernel(env_t env_t_ptr, program_t program_ptr,
                     const char *kernel_name, kernel_t *kernel_ptr);

DPPL_API
int destroy_dp_kernel(kernel_t *kernel_ptr);

/*!
 *
 */
DPPL_API
int create_dp_kernel_arg(const void *arg_value, size_t arg_size,
                         kernel_arg_t *kernel_arg_t_ptr);

/*!
 *
 */
DPPL_API
int create_dp_kernel_arg_from_buffer(buffer_t *buffer_t_ptr,
                                     kernel_arg_t *kernel_arg_t_ptr);

/*!
 *
 */
DPPL_API
int destroy_dp_kernel_arg(kernel_arg_t *kernel_arg_t_ptr);

/*!
 *
 */
DPPL_API
int set_args_and_enqueue_dp_kernel(env_t env_t_ptr, kernel_t kernel_t_ptr,
                                   size_t nargs, const kernel_arg_t *args,
                                   unsigned int work_dim,
                                   const size_t *global_work_offset,
                                   const size_t *global_work_size,
                                   const size_t *local_work_size);

/*!
 *
 */
DPPL_API
int set_args_and_enqueue_dp_kernel_auto_blocking(
    env_t env_t_ptr, kernel_t kernel_t_ptr, size_t nargs,
    const kernel_arg_t *args, unsigned int num_dims, size_t *dim_starts,
    size_t *dim_stops);

/*!
 *
 */
DPPL_API
int retain_dp_context(env_t env_t_ptr);

/*!
 *
 */
DPPL_API
int release_dp_context(env_t env_t_ptr);

//---- TODO:

// 1. Add release/retain methods for buffers

//---------
