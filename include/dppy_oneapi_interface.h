//===-- dppy_oneapi_interface.h - DPPY-SYCL interface ---*- C++ -*---------===//
//
//                     Data Parallel Python (DPPY)
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of a C API to expose a lightweight SYCL
/// interface for the Python dppy package.
///
//===----------------------------------------------------------------------===//

#ifndef DPPY_ONEAPI_INTERFACE_H_
#define DPPY_ONEAPI_INTERFACE_H_

#include <stdbool.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////////////////////////
/////////////////////////// oneapi_interface_mem_buffer ////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*!
 *
 */
struct oneapi_interface_mem_buffer
{
    unsigned int id_;
    // A cl_mem pointer
    void         *buffer_ptr;
    // Stores the size of the mem_buffer_ptr (e.g sizeof(cl_mem))
    size_t       sizeof_mem_buffer_ptr;
};

typedef struct oneapi_interface_mem_buffer* mem_buffer_t;

////////////////////////////////////////////////////////////////////////////////
/////////////////////////// oneapi_interface_mem_usm ///////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*!
 *
 */
struct oneapi_interface_mem_usm
{
    unsigned int id_;
    void         *usm_ptr;
};

typedef struct oneapi_interface_mem_usm* mem_usm_t;

////////////////////////////////////////////////////////////////////////////////
////////////////////////// oneapi_interface_device_env /////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*!
 *
 */
struct oneapi_interface_device_env
{
    unsigned int id_;
    void         *sycl_context;
    void         *sycl_device;
    void         *sycl_queue;
    void         *ocl_context;
    void         *ocl_device;
    void         *ocl_queue;
    unsigned int max_work_item_dims;
    unsigned int max_work_group_size;

    int (*dump_fn) (void *);
};

typedef struct oneapi_interface_device_env* env_t;


/*!
 * @brief Free the env_t object and all its resources.
 *
 * @param[in] e - Pointer to the env_t object to be freed
 *
 * @return An error code indicating if resource freeing was successful.
 */
int destroy_oneapi_interface_device_env (env_t *e);

////////////////////////////////////////////////////////////////////////////////
//////////////////////////// oneapi_interface_runtime //////////////////////////
////////////////////////////////////////////////////////////////////////////////


/*! @struct oneapi_interface_runtime
 *  @brief Stores an array of the available dp_env objects.
 *
 *  @var oneapi_interface_runtime::num_platforms
 *  The number of available dp_env objects
 *
 */
struct oneapi_interface_runtime
{
    unsigned int id_;
    unsigned int num_platforms;
    bool         has_cpu;
    bool         has_gpu;
    unsigned int num_cpus;
    unsigned int num_gpus;

    int (*get_default_env) (env_t *);
    int (*get_gpu_env)     (env_t *, int);
    int (*get_cpu_env)     (env_t *, int);

    int (*dump_fn)         (void *);
};

typedef struct oneapi_interface_runtime* runtime_t;


/*!
 * @brief Initializes a new runtime_t object
 *
 * @param[in/out] rt - An uninitialized runtime_t pointer that is initialized
 *                     by the function.
 *
 * @return An error code indicating if the runtime_t object was successfully
 *         initialized.
 */
int create_oneapi_interface_runtime (runtime_t *rt);


/*!
 * @brief Free the runtime_t object and all its resources.
 *
 * @param[in] rt - Pointer to the runtime_t object to be freed
 *
 * @return An error code indicating if resource freeing was successful.
 */
int destroy_oneapi_interface_runtime (runtime_t *rt);


#endif /*--- DPPY_ONEAPI_INTERFACE_H_ ---*/
