//===------ dppl_opencl_interface.c - DPPL-OpenCL interface ----*- C -*----===//
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
/// This file implements the data types and functions declared in
/// dppl_opencl_interface.h.
///
//===----------------------------------------------------------------------===//
#include "dppl_opencl_interface.h"
#include "error_check_macros.h"
#include <string.h>
#include <assert.h>
#include <CL/cl.h>  /* OpenCL headers */

/*------------------------------- Magic numbers ------------------------------*/

#define RUNTIME_ID   0x6dd5e8c8
#define ENV_ID       0x6c78fd87
#define BUFFER_ID    0xc55c47b1
#define KERNEL_ID    0x032dc08e
#define PROGRAM_ID   0xc3842d12
#define KERNELARG_ID 0xd42f630f

#if DEBUG

static void check_runtime_id (runtime_t x)
{
    assert(x->id_ == RUNTIME_ID);
}

static void check_env_id (env_t x)
{
    assert(x->id_ == ENV_ID);
}

static void check_buffer_id (buffer_t x)
{
    assert(x->id_ == BUFFER_ID);
}

static void check_kernel_id (kernel_t x)
{
    assert(x->id_ == KERNEL_ID);
}

static void check_program_id (program_t x)
{
    assert(x->id_ == PROGRAM_ID);
}

static void check_kernelarg_id (kernel_arg_t x)
{
    assert(x->id_ == KERNELARG_ID);
}

#endif

/*------------------------------- Private helpers ----------------------------*/


static int get_platform_name (cl_platform_id platform, char **platform_name)
{
    cl_int err;
    size_t n;

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, *platform_name, &n);
    CHECK_OPEN_CL_ERROR(err, "Could not get platform name length.");

    // Allocate memory for the platform name string
    *platform_name = (char*)malloc(sizeof(char)*n);
    CHECK_MALLOC_ERROR(char*, *platform_name);

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, n, *platform_name,
            NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get platform name.");

    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(*platform_name);
    return DP_GLUE_FAILURE;
}


/*!
 *
 */
static int dump_device_info (void *obj)
{
    cl_int err;
    char *value;
    size_t size;
    cl_uint maxComputeUnits;
    env_t env_t_ptr;

    value = NULL;
    env_t_ptr = (env_t)obj;
    cl_device_id device = (cl_device_id)(env_t_ptr->device);

    err = clRetainDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not retain device.");

    // print device name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get device name.");
    value = (char*)malloc(size);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get device name.");
    printf("Device: %s\n", value);
    free(value);

    // print hardware device version
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get device version.");
    value = (char*) malloc(size);
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get device version.");
    printf("Hardware version: %s\n", value);
    free(value);

    // print software driver version
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get driver version.");
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get driver version.");
    printf("Software version: %s\n", value);
    free(value);

    // print c version supported by compiler for device
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get open cl version.");
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get open cl version.");
    printf("OpenCL C version: %s\n", value);
    free(value);

    // print parallel compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get number of compute units.");
    printf("Parallel compute units: %d\n", maxComputeUnits);

    err = clReleaseDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not release device.");

    return DP_GLUE_SUCCESS;

error:
    free(value);
    return DP_GLUE_FAILURE;
}


/*!
 * @brief Helper function to print out information about the platform and
 * devices available to this runtime.
 *
 */
static int dump_dp_runtime_info (void *obj)
{
    size_t i;
    runtime_t rt;

    rt = (runtime_t)obj;
#if DEBUG
    check_runtime_id(rt);
#endif
    if(rt) {
        printf("Number of platforms : %d\n", rt->num_platforms);
        cl_platform_id *platforms = rt->platform_ids;
        for(i = 0; i < rt->num_platforms; ++i) {
            char *platform_name = NULL;
            get_platform_name(platforms[i], &platform_name);
            printf("Platform #%zu: %s\n", i, platform_name);
            free(platform_name);
        }
    }

    return DP_GLUE_SUCCESS;
}


/*!
 *
 */
static int dump_dp_kernel_info (void *obj)
{
    cl_int err;
    char *value;
    size_t size;
    cl_uint numKernelArgs;
    cl_kernel kernel;
    kernel_t kernel_t_ptr;

    value = NULL;
    kernel_t_ptr = (kernel_t)obj;
#if DEBUG
    check_kernel_id(kernel_t_ptr);
#endif
    kernel = (cl_kernel)(kernel_t_ptr->kernel);

    // print kernel function name
    err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &size);
    CHECK_OPEN_CL_ERROR(err, "Could not get kernel function name size.");
    value = (char*)malloc(size);
    err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, size, value, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get kernel function name.");
    printf("Kernel Function name: %s\n", value);
    free(value);

    // print the number of kernel args
    err = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(numKernelArgs),
            &numKernelArgs, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get kernel num args.");
    printf("Number of kernel arguments : %d\n", numKernelArgs);

    return DP_GLUE_SUCCESS;

error:
    free(value);
    return DP_GLUE_FAILURE;
}


/*!
 *
 */
static int get_first_device (cl_platform_id* platforms,
                             cl_uint platformCount,
                             cl_device_id *device,
                             cl_device_type device_ty)
{
    cl_int status;
    cl_uint ndevices = 0;
    unsigned int i;

    for (i = 0; i < platformCount; ++i) {
        // get all devices of env_ty
        status = clGetDeviceIDs(platforms[i], device_ty, 0, NULL, &ndevices);
        // If this platform has no devices of this type then continue
        if(!ndevices) continue;

        // get the first device
        status = clGetDeviceIDs(platforms[i], device_ty, 1, device, NULL);
        CHECK_OPEN_CL_ERROR(status, "Could not get first cl_device_id.");

        // If the first device of this type was discovered, no need to look more
        if(ndevices) break;
    }

    if(ndevices)
        return DP_GLUE_SUCCESS;
    else
        return DP_GLUE_FAILURE;

error:
    return DP_GLUE_FAILURE;
}

static int support_int64_atomics(cl_device_id *device)
{

    cl_int err;
    size_t size;
    char *value;

    err = clGetDeviceInfo(*device, CL_DEVICE_EXTENSIONS, 0, NULL, &size);
    if (err != CL_SUCCESS ) {
        printf("Unable to obtain device info for param\n");
        return DP_GLUE_FAILURE;
    }
    value = (char*) malloc(size);
    clGetDeviceInfo(*device, CL_DEVICE_EXTENSIONS, size, value, NULL);

    if(strstr(value, "cl_khr_int64_base_atomics") != NULL) {
        return DP_GLUE_SUCCESS;
    } else {
        return DP_GLUE_FAILURE;
    }
}

static int support_float64_atomics(cl_device_id *device)
{

    cl_int err;
    size_t size;
    char *value;

    err = clGetDeviceInfo(*device, CL_DEVICE_EXTENSIONS, 0, NULL, &size);
    if (err != CL_SUCCESS ) {
        printf("Unable to obtain device info for param\n");
        return DP_GLUE_FAILURE;
    }
    value = (char*) malloc(size);
    clGetDeviceInfo(*device, CL_DEVICE_EXTENSIONS, size, value, NULL);

    if(strstr(value, "cl_khr_fp64") != NULL) {
        return DP_GLUE_SUCCESS;
    } else {
        return DP_GLUE_FAILURE;
    }
}

/*!
 *
 */
static int create_dp_env_t (cl_platform_id* platforms,
                            size_t nplatforms,
                            cl_device_type device_ty,
                            env_t *env_t_ptr)
{
    cl_int err;
    int err1;
    env_t env;
    cl_device_id *device;

    env = NULL;
    device = NULL;

    // Allocate the env_t object
    env = (env_t)malloc(sizeof(struct dp_env));
    CHECK_MALLOC_ERROR(env_t, env);
    env->id_ = ENV_ID;

    env->context = NULL;
    env->device = NULL;
    env->queue = NULL;
    env->max_work_item_dims = 0;
    env->max_work_group_size = 0;
    env->dump_fn = NULL;

    device = (cl_device_id*)malloc(sizeof(cl_device_id));

    err1 = get_first_device(platforms, nplatforms, device, device_ty);
    CHECK_DPGLUE_ERROR(err1, "Failed inside get_first_device");

    // get the CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS for this device
    err = clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
            sizeof(env->max_work_item_dims), &env->max_work_item_dims, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get max work item dims");

    // get the CL_DEVICE_MAX_WORK_GROUP_SIZE for this device
    err = clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(env->max_work_group_size), &env->max_work_group_size, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not get max work group size");

    // Create a context and associate it with device
    env->context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create device context.");
    // Create a queue and associate it with the context
    env->queue = clCreateCommandQueueWithProperties((cl_context)env->context,
            *device, 0, &err);

    CHECK_OPEN_CL_ERROR(err, "Could not create command queue.");

    env->device = *device;
    env ->dump_fn = dump_device_info;

    if (DP_GLUE_SUCCESS == support_int64_atomics(device)) {
        env->support_int64_atomics = 1;
    } else {
        env->support_int64_atomics = 0;
    }

    if (DP_GLUE_SUCCESS == support_float64_atomics(device)) {
        env->support_float64_atomics = 1;
    } else {
        env->support_float64_atomics = 0;
    }

    free(device);
    *env_t_ptr = env;

    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(env);
    *env_t_ptr = NULL;
    return DP_GLUE_FAILURE;
}


static int destroy_dp_env_t (env_t *env_t_ptr)
{
    cl_int err;
#if DEBUG
    check_env_id(*env_t_ptr);
#endif
    err = clReleaseCommandQueue((cl_command_queue)(*env_t_ptr)->queue);
    CHECK_OPEN_CL_ERROR(err, "Could not release command queue.");
    err = clReleaseDevice((cl_device_id)(*env_t_ptr)->device);
    CHECK_OPEN_CL_ERROR(err, "Could not release device.");
    err = clReleaseContext((cl_context)(*env_t_ptr)->context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context.");

    free(*env_t_ptr);

    return DP_GLUE_SUCCESS;

error:
    return DP_GLUE_FAILURE;
}


/*!
 * @brief Initialize the runtime object.
 */
static int init_runtime_t_obj (runtime_t rt)
{
    cl_int status;
    int ret;
    cl_platform_id *platforms;
#if DEBUG
    check_runtime_id(rt);
#endif
    // get count of available platforms
    status = clGetPlatformIDs(0, NULL, &(rt->num_platforms));
    CHECK_OPEN_CL_ERROR(status, "Could not get platform count.");

    if(!rt->num_platforms) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        goto error;
    }

    // Allocate memory for the platforms array
    rt->platform_ids = (cl_platform_id*)malloc(
                           sizeof(cl_platform_id)*rt->num_platforms
                       );
    CHECK_MALLOC_ERROR(cl_platform_id, rt->platform_ids);

    // Get the platforms
    status = clGetPlatformIDs(rt->num_platforms, rt->platform_ids, NULL);
    CHECK_OPEN_CL_ERROR(status, "Could not get platform ids");
    // Cast rt->platforms to a pointer of type cl_platform_id, as we cannot do
    // pointer arithmetic on void*.
    platforms = (cl_platform_id*)rt->platform_ids;
    // Get the first cpu device on this platform
    ret = create_dp_env_t(platforms, rt->num_platforms,
                          CL_DEVICE_TYPE_CPU, &rt->first_cpu_env);
    rt->has_cpu = !ret;

#if DEBUG
    if(rt->has_cpu)
        printf("DEBUG: CPU device acquired...\n");
    else
        printf("DEBUG: No CPU available on the system\n");
#endif

    // Get the first gpu device on this platform
    ret = create_dp_env_t(platforms, rt->num_platforms,
                          CL_DEVICE_TYPE_GPU, &rt->first_gpu_env);
    rt->has_gpu = !ret;

#if DEBUG
    if(rt->has_gpu)
        printf("DEBUG: GPU device acquired...\n");
    else
        printf("DEBUG: No GPU available on the system.\n");
#endif

    if(rt->has_gpu)
        rt->curr_env = rt->first_gpu_env;
    else if(rt->has_cpu)
        rt->curr_env = rt->first_cpu_env;
    else
        goto error;

    return DP_GLUE_SUCCESS;

malloc_error:

    return DP_GLUE_FAILURE;
error:
    free(rt->platform_ids);

    return DP_GLUE_FAILURE;
}

/*-------------------------- End of private helpers --------------------------*/

int set_curr_env (runtime_t rt, env_t env)
{
    if(env && rt) {
        rt->curr_env = env;
        return DP_GLUE_SUCCESS;
    }
    return DP_GLUE_FAILURE;
}

/*!
 * @brief Initializes a new dp_runtime_t object
 *
 */
int create_dp_runtime (runtime_t *rt)
{
    int err;
    runtime_t rtobj;

    rtobj = NULL;
    // Allocate a new struct dp_runtime object
    rtobj = (runtime_t)malloc(sizeof(struct dp_runtime));
    CHECK_MALLOC_ERROR(runtime_t, rt);

    rtobj->id_ = RUNTIME_ID;
    rtobj->num_platforms = 0;
    rtobj->platform_ids  = NULL;
    err = init_runtime_t_obj(rtobj);
    CHECK_DPGLUE_ERROR(err, "Could not initialize runtime object.");
    rtobj->dump_fn = dump_dp_runtime_info;

    *rt = rtobj;
#if DEBUG
    printf("DEBUG: Created an new dp_runtime object\n");
#endif
    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(rtobj);
    return DP_GLUE_FAILURE;
}


/*!
 * @brief Free the runtime and all its resources.
 *
 */
int destroy_dp_runtime (runtime_t *rt)
{
    int err;
#if DEBUG
    check_runtime_id(*rt);
#endif

#if DEBUG
    printf("DEBUG: Going to destroy the dp_runtime object\n");
#endif
    // free the first_cpu_device
    if((*rt)->first_cpu_env) {
        err = destroy_dp_env_t(&(*rt)->first_cpu_env);
        CHECK_DPGLUE_ERROR(err, "Could not destroy first_cpu_device.");
    }

    // free the first_gpu_device
    if((*rt)->first_gpu_env) {
        err = destroy_dp_env_t(&(*rt)->first_gpu_env);
        CHECK_DPGLUE_ERROR(err, "Could not destroy first_gpu_device.");
    }

    // free the platforms
    free((cl_platform_id*)(*rt)->platform_ids);
    // free the runtime_t object
    free(*rt);

#if DEBUG
    printf("DEBUG: Destroyed the new dp_runtime object\n");
#endif
    return DP_GLUE_SUCCESS;

error:
    return DP_GLUE_FAILURE;
}


/*!
 *
 */
int retain_dp_context (env_t env_t_ptr)
{
    cl_int err;
    cl_context context;
#if DEBUG
    check_env_id(env_t_ptr);
#endif
    context = (cl_context)(env_t_ptr->context);
    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed when calling clRetainContext.");

    return DP_GLUE_SUCCESS;
error:
    return DP_GLUE_FAILURE;
}


/*!
 *
 */
int release_dp_context (env_t env_t_ptr)
{
    cl_int err;
    cl_context context;
#if DEBUG
    check_env_id(env_t_ptr);
#endif
    context = (cl_context)(env_t_ptr->context);
    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed when calling clRetainContext.");

    return DP_GLUE_SUCCESS;
error:
    return DP_GLUE_FAILURE;
}


int create_dp_rw_mem_buffer (env_t env_t_ptr,
                             size_t buffsize,
                             buffer_t *buffer_t_ptr)
{
    cl_int err;
    buffer_t buff;
    cl_context context;
#if DEBUG
    check_env_id(env_t_ptr);
#endif
    buff = NULL;

    // Get the context from the device
    context = (cl_context)(env_t_ptr->context);
    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain context.");

    // Allocate a dp_buffer object
    buff = (buffer_t)malloc(sizeof(struct dp_buffer));
    CHECK_MALLOC_ERROR(buffer_t, buffer_t_ptr);

    buff->id_ = BUFFER_ID;

    // Create the OpenCL buffer.
    // NOTE : Copying of data from host to device needs to happen explicitly
    // using clEnqueue[Write|Read]Buffer. This would change in the future.
    buff->buffer_ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, buffsize,
                                      NULL, &err);
    CHECK_OPEN_CL_ERROR(err, "Failed to create CL buffer.");

    buff->sizeof_buffer_ptr = sizeof(cl_mem);
#if DEBUG
    printf("DEBUG: CL RW buffer created...\n");
#endif
    *buffer_t_ptr = buff;
    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Failed to release context.");

    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(buff);
    return DP_GLUE_FAILURE;
}


int destroy_dp_rw_mem_buffer (buffer_t *buff)
{
    cl_int err;
#if DEBUG
    check_buffer_id(*buff);
#endif
    err = clReleaseMemObject((cl_mem)(*buff)->buffer_ptr);
    CHECK_OPEN_CL_ERROR(err, "Failed to release CL buffer.");
    free(*buff);

#if DEBUG
    printf("DEBUG: CL buffer destroyed...\n");
#endif

    return DP_GLUE_SUCCESS;

error:
    return DP_GLUE_FAILURE;
}


int write_dp_mem_buffer_to_device (env_t env_t_ptr,
                                   buffer_t buffer_t_ptr,
                                   bool blocking,
                                   size_t offset,
                                   size_t buffersize,
                                   const void *data_ptr)
{
    cl_int err;
    cl_command_queue queue;
    cl_mem mem;
#if DEBUG
    check_env_id(env_t_ptr);
    check_buffer_id(buffer_t_ptr);
#endif
    queue = (cl_command_queue)env_t_ptr->queue;
    mem = (cl_mem)buffer_t_ptr->buffer_ptr;

#if DEBUG
    assert(mem && "buffer memory is NULL");
#endif

    err = clRetainMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");
    err = clRetainCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the buffer memory object.");

    // Not using any events for the time being. Eventually we want to figure
    // out the event dependencies using parfor analysis.
    err = clEnqueueWriteBuffer(queue, mem, blocking?CL_TRUE:CL_FALSE,
            offset, buffersize, data_ptr, 0, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Failed to write to CL buffer.");

    err = clReleaseCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the command queue.");
    err = clReleaseMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the buffer memory object.");

#if DEBUG
    printf("DEBUG: CL buffer written to device...\n");
#endif
    //--- TODO: Implement a version that uses clEnqueueMapBuffer

    return DP_GLUE_SUCCESS;
error:
    return DP_GLUE_FAILURE;
}


int read_dp_mem_buffer_from_device (env_t env_t_ptr,
                                    buffer_t buffer_t_ptr,
                                    bool blocking,
                                    size_t offset,
                                    size_t buffersize,
                                    void *data_ptr)
{
    cl_int err;
    cl_command_queue queue;
    cl_mem mem;
#if DEBUG
    check_env_id(env_t_ptr);
    check_buffer_id(buffer_t_ptr);
#endif
    queue = (cl_command_queue)env_t_ptr->queue;
    mem = (cl_mem)buffer_t_ptr->buffer_ptr;

    err = clRetainMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");
    err = clRetainCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to retain the command queue.");

    // Not using any events for the time being. Eventually we want to figure
    // out the event dependencies using parfor analysis.
    err = clEnqueueReadBuffer(queue, mem, blocking?CL_TRUE:CL_FALSE,
            offset, buffersize, data_ptr, 0, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Failed to read from CL buffer.");

    err = clReleaseCommandQueue(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the command queue.");
    err = clReleaseMemObject(mem);
    CHECK_OPEN_CL_ERROR(err, "Failed to release the buffer memory object.");

#if DEBUG
    printf("DEBUG: CL buffer read from device...\n");
#endif
    fflush(stdout);
    //--- TODO: Implement a version that uses clEnqueueMapBuffer

    return DP_GLUE_SUCCESS;
error:
    return DP_GLUE_FAILURE;
}


int create_dp_program_from_spirv (env_t env_t_ptr,
                                  const void *il,
                                  size_t length,
                                  program_t *program_t_ptr)
{
    cl_int err;
    cl_context context;
    program_t prog;
#if DUMP_SPIRV
    FILE *write_file;
#endif
#if DEBUG
    check_env_id(env_t_ptr);
#endif
    prog = NULL;

#if DUMP_SPIRV
    write_file = fopen("latest.spirv","wb");
    fwrite(il,length,1,write_file);
    fclose(write_file);
#endif

    prog = (program_t)malloc(sizeof(struct dp_program));
    CHECK_MALLOC_ERROR(program_t, program_t_ptr);

    prog->id_ = PROGRAM_ID;

    context = (cl_context)env_t_ptr->context;

    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not retain context");
    // Create a program with a SPIR-V file
    prog->program = clCreateProgramWithIL(context, il, length, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create program with IL");
#if DEBUG
    printf("DEBUG: CL program created from spirv of length %zu...\n", length);
#endif

    *program_t_ptr = prog;

    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context");

    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(prog);
    return DP_GLUE_FAILURE;
}


int create_dp_program_from_source (env_t env_t_ptr,
                                   unsigned int count,
                                   const char **strings,
                                   const size_t *lengths,
                                   program_t *program_t_ptr)
{
    cl_int err;
    cl_context context;
    program_t prog;
#if DEBUG
    check_env_id(env_t_ptr);
#endif
    prog = NULL;
    prog = (program_t)malloc(sizeof(struct dp_program));
    CHECK_MALLOC_ERROR(program_t, program_t_ptr);

    prog->id_ = PROGRAM_ID;

    context = (cl_context)env_t_ptr->context;

    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not retain context");
    // Create a program with string source files
    prog->program = clCreateProgramWithSource(context, count, strings,
            lengths, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create program with source");
#if DEBUG
    printf("DEBUG: CL program created from source...\n");
#endif

    *program_t_ptr = prog;

    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context");

    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(prog);
    return DP_GLUE_FAILURE;
}


int destroy_dp_program (program_t *program_ptr)
{
    cl_int err;
#if DEBUG
    check_program_id(*program_ptr);
#endif
    err = clReleaseProgram((cl_program)(*program_ptr)->program);
    CHECK_OPEN_CL_ERROR(err, "Failed to release CL program.");
    free(*program_ptr);

#if DEBUG
    printf("DEBUG: CL program destroyed...\n");
#endif

    return DP_GLUE_SUCCESS;

error:
    return DP_GLUE_FAILURE;
}


int build_dp_program (env_t env_t_ptr, program_t program_t_ptr)
{
    cl_int err;
    cl_device_id device;
#if DEBUG
    check_env_id(env_t_ptr);
    check_program_id(program_t_ptr);
#endif
    device = (cl_device_id)env_t_ptr->device;
    err = clRetainDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not retain device");
    // Build (compile) the program for the device
    err = clBuildProgram((cl_program)program_t_ptr->program, 1, &device, NULL,
            NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not build program");
#if DEBUG
    printf("DEBUG: CL program successfully built.\n");
#endif
    err = clReleaseDevice(device);
    CHECK_OPEN_CL_ERROR(err, "Could not release device");

    return DP_GLUE_SUCCESS;

error:
    return DP_GLUE_FAILURE;
}


/*!
 *
 */
int create_dp_kernel (env_t env_t_ptr,
                      program_t program_t_ptr,
                      const char *kernel_name,
                      kernel_t *kernel_ptr)
{
    cl_int err;
    cl_context context;
    kernel_t ker;
#if DEBUG
    check_env_id(env_t_ptr);
#endif
    ker = NULL;
    ker = (kernel_t)malloc(sizeof(struct dp_kernel));
    CHECK_MALLOC_ERROR(kernel_t, kernel_ptr);

    ker->id_ = KERNEL_ID;

    context = (cl_context)(env_t_ptr->context);
    err = clRetainContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not retain context");
    ker->kernel = clCreateKernel((cl_program)(program_t_ptr->program),
            kernel_name, &err);
    CHECK_OPEN_CL_ERROR(err, "Could not create kernel");
    err = clReleaseContext(context);
    CHECK_OPEN_CL_ERROR(err, "Could not release context");
#if DEBUG
    printf("DEBUG: CL kernel created\n");
#endif
    ker->dump_fn = dump_dp_kernel_info;
    *kernel_ptr = ker;
    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
error:
    free(ker);
    return DP_GLUE_FAILURE;
}


int destroy_dp_kernel (kernel_t *kernel_ptr)
{
    cl_int err;
#if DEBUG
    check_kernel_id(*kernel_ptr);
#endif
    err = clReleaseKernel((cl_kernel)(*kernel_ptr)->kernel);
    CHECK_OPEN_CL_ERROR(err, "Failed to release CL kernel.");
    free(*kernel_ptr);

#if DEBUG
    printf("DEBUG: CL kernel destroyed...\n");
#endif

    return DP_GLUE_SUCCESS;

error:
    return DP_GLUE_FAILURE;
}


/*!
 *
 */
int create_dp_kernel_arg (const void *arg_value,
                          size_t arg_size,
                          kernel_arg_t *kernel_arg_t_ptr)
{
    kernel_arg_t kernel_arg;

    kernel_arg = NULL;
    kernel_arg = (kernel_arg_t)malloc(sizeof(struct dp_kernel_arg));
    CHECK_MALLOC_ERROR(kernel_arg_t, kernel_arg);

    kernel_arg->id_ = KERNELARG_ID;
    kernel_arg->arg_size = arg_size;
    kernel_arg->arg_value = arg_value;

#if DEBUG
    printf("DEBUG: Kernel arg created\n");
//    void **tp = (void**)kernel_arg->arg_value;
//    printf("DEBUG: create_kernel_arg %p (size %ld, addr %p)\n",
//            kernel_arg, kernel_arg->arg_size, *tp);
#endif

    *kernel_arg_t_ptr = kernel_arg;

    return DP_GLUE_SUCCESS;

malloc_error:
    return DP_GLUE_FAILURE;
}

/*!
 *
 */
int create_dp_kernel_arg_from_buffer (buffer_t *buffer_t_ptr,
                                      kernel_arg_t *kernel_arg_t_ptr)
{
#if DEBUG
    check_buffer_id(*buffer_t_ptr);
#endif
    return create_dp_kernel_arg(&((*buffer_t_ptr)->buffer_ptr),
                                (*buffer_t_ptr)->sizeof_buffer_ptr,
                                kernel_arg_t_ptr);
}

/*!
 *
 */
int destroy_dp_kernel_arg (kernel_arg_t *kernel_arg_t_ptr)
{
    free(*kernel_arg_t_ptr);

#if DEBUG
    printf("DEBUG: Kernel arg destroyed...\n");
#endif

    return DP_GLUE_SUCCESS;
}


/*!
 *
 */
int set_args_and_enqueue_dp_kernel (env_t env_t_ptr,
                                    kernel_t kernel_t_ptr,
                                    size_t nargs,
                                    const kernel_arg_t *array_of_args,
                                    unsigned int work_dim,
                                    const size_t *global_work_offset,
                                    const size_t *global_work_size,
                                    const size_t *local_work_size)
{
    size_t i;
    cl_int err;
    cl_kernel kernel;
    cl_command_queue queue;

    err = 0;
#if DEBUG
    check_env_id(env_t_ptr);
    check_kernel_id(kernel_t_ptr);
#endif
    kernel = (cl_kernel)kernel_t_ptr->kernel;
    queue = (cl_command_queue)env_t_ptr->queue;
#if DEBUG
    kernel_t_ptr->dump_fn(kernel_t_ptr);
#endif
    // Set the kernel arguments
    for(i = 0; i < nargs; ++i) {
#if DEBUG
        printf("DEBUG: clSetKernelArgs for arg # %zu\n", i);
#endif
        kernel_arg_t this_arg = array_of_args[i];
#if DEBUG
        check_kernelarg_id(this_arg);
        void **tp = (void**)this_arg->arg_value;
        printf("DEBUG: clSetKernelArgs for arg # %zu (size %zu, addr %p)\n", i,
                this_arg->arg_size, *tp);
#endif
        err = clSetKernelArg(kernel, i, this_arg->arg_size,
                             this_arg->arg_value);
        CHECK_OPEN_CL_ERROR(err, "Could not set arguments to the kernel");
    }

    // Execute the kernel. Not using events for the time being.
    err = clEnqueueNDRangeKernel(queue, kernel, work_dim, global_work_offset,
            global_work_size, local_work_size, 0, NULL, NULL);
    CHECK_OPEN_CL_ERROR(err, "Could not enqueue the kernel");

    err = clFinish(queue);
    CHECK_OPEN_CL_ERROR(err, "Failed while waiting for queue to finish");
#if DEBUG
    printf("DEBUG: CL Kernel Finish...\n");
#endif
    return DP_GLUE_SUCCESS;

error:
    return DP_GLUE_FAILURE;
}


/*!
 *
 */
int set_args_and_enqueue_dp_kernel_auto_blocking (env_t env_t_ptr,
                                                  kernel_t kernel_t_ptr,
                                                  size_t nargs,
                                                  const kernel_arg_t *args,
                                                  unsigned int num_dims,
                                                  size_t *dim_starts,
                                                  size_t *dim_stops)
{
    size_t *global_work_size;
//    size_t *local_work_size;
    int err;
    unsigned i;

    global_work_size = (size_t*)malloc(sizeof(size_t) * num_dims);
//    local_work_size  = (size_t*)malloc(sizeof(size_t) * num_dims);
    CHECK_MALLOC_ERROR(size_t, global_work_size);
//    CHECK_MALLOC_ERROR(size_t, local_work_size);

    assert(num_dims > 0 && num_dims < 4);
    for (i = 0; i < num_dims; ++i) {
        global_work_size[i] = dim_stops[i] - dim_starts[i] + 1;
    }

    err = set_args_and_enqueue_dp_kernel(env_t_ptr,
                                         kernel_t_ptr,
                                         nargs,
                                         args,
                                         num_dims,
                                         NULL,
                                         global_work_size,
                                         NULL);
    free(global_work_size);
//    free(local_work_size);
    return err;

malloc_error:
    free(global_work_size);
//    free(local_work_size);
    return DP_GLUE_FAILURE;
}
