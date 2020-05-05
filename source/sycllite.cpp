#ifdef __cplusplus
extern "C" {
#endif
#include "dppy_oneapi_interface.h"
#ifdef __cplusplus
}
#endif

#include "error_check_macros.h"

#include <cassert>
#include <CL/sycl.hpp>  /* SYCL headers */
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace cl::sycl;

/*------------------------------- Private helpers ----------------------------*/

// Anonymous namespace for private helpers
namespace
{

constexpr size_t RUNTIME_ID   = 0x6dd5e8c8;
constexpr size_t ENV_ID       = 0x6c78fd87;
constexpr size_t BUFFER_ID    = 0xc55c47b1;
constexpr size_t KERNEL_ID    = 0x032dc08e;
constexpr size_t PROGRAM_ID   = 0xc3842d12;
constexpr size_t KERNELARG_ID = 0xd42f630f;

enum ONEAPI_INTERFACE_ERROR_CODES
{
    OAI_SUCCESS = 0,
    OAI_FAILURE = -1
};

int dump_oneapi_interface_device_env (env_t Env)
{
    device *Device;
    std::stringstream ss;

    Device = static_cast<device*>(Env->sycl_device);

    ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
       << Device->get_info<info::device::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Driver version"
       << Device->get_info<info::device::driver_version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
       << Device->get_info<info::device::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Profile"
       << Device->get_info<info::device::profile>() << '\n';


    std::cout << ss.str();

    return OAI_SUCCESS;
}

void dump_platform_info (const platform & Platform)
{
    std::stringstream ss;

    ss << std::setw(4) << " " << std::left << std::setw(12) << "Name"
       << Platform.get_info<info::platform::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Version"
       << Platform.get_info<info::platform::version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Vendor"
       << Platform.get_info<info::platform::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(12) << "Profile"
       << Platform.get_info<info::platform::profile>() << '\n';

    std::cout << ss.str();
}

/*!
 * @brief Prints out the metadata of a oneapi_interface_runtime object
 *
 */
int dump_oneapi_interface_runtime (void *RuntimeObj)
{
    size_t i = 0;
    auto rt = static_cast<runtime_t>(RuntimeObj);

    // Print out the info for each platform
    auto platforms = platform::get_platforms();
    for (auto &p : platforms) {
        std::cout << "Platform " << i << '\n';
        dump_platform_info(p);
        ++i;
    }

    // Print out the info for CPU devices
    if(rt->has_cpu)
        std::cout << "Number of available OpenCL CPU devices: "
                  << rt->num_cpus << '\n';

    // Print out the info for GPU devices
    if(rt->has_gpu)
        std::cout << "Number of available OpenCL GPU devices: "
                  << rt->num_gpus << '\n';

    return OAI_SUCCESS;
}


void create_create_oneapi_interface_env (env_t *Env, device Device)
{
    // TODO

}


void init_runtime (runtime_t Runtime)
{
    Runtime->id_ = RUNTIME_ID;
    Runtime->num_platforms = platform::get_platforms().size();

    auto gpu_devices = device::get_devices(info::device_type::gpu);
    auto cpu_devices = device::get_devices(info::device_type::cpu);

    if(!gpu_devices.empty()) {
        Runtime->has_gpu = true;
        Runtime->num_gpus = gpu_devices.size();
    }

    if(!cpu_devices.empty()) {
        Runtime->has_cpu = true;
        Runtime->num_cpus = cpu_devices.size();
    }

    // Create the env constructors

    // Set dump_fn
    Runtime->dump_fn = dump_oneapi_interface_runtime;
}

}

/*------------------------------- Public API ---------------------------------*/

/*!
 * @brief Initializes a new dp_runtime_t object
 *
 */
int create_oneapi_interface_runtime (runtime_t *rt)
{
    // Allocate the runtime_t object
    *rt = new oneapi_interface_runtime;
    // Initialize the runtime_t object
    init_runtime(*rt);

    return OAI_SUCCESS;
}


/*!
 * @brief Free the runtime and all its resources.
 *
 */
int destroy_oneapi_interface_runtime (runtime_t *rt)
{

#if DEBUG
    check_runtime_id(*rt);
#endif

#if DEBUG
    std::err << "DEBUG: Going to destroy the dp_runtime object\n";
#endif
    // free cpu_envs
    // TODO

    // free gpu_envs
    // TODO

    // free the runtime_t object
    delete *rt;

#if DEBUG
    std::err << "DEBUG: Destroyed the new dp_runtime object\n";
#endif

    return OAI_SUCCESS;
}
