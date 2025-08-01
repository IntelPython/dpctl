//===-------- dpctl_sycl_device_manager.cpp - helpers for sycl devices ------=//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2025 Intel Corporation
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
/// This file implements the functions declared in dpctl_sycl_device_manager.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_manager.h"
#include "dpctl_error_handlers.h"
#include "dpctl_string_utils.hpp"
#include "dpctl_sycl_enum_types.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_utils_helper.h"
#include <Config/dpctl_config.h> /* Config */
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stddef.h>
#include <sycl/sycl.hpp> /* SYCL headers   */
#include <unordered_map>
#include <utility>
#include <vector>

using namespace sycl;

namespace
{

static_assert(__SYCL_COMPILER_VERSION >= __SYCL_COMPILER_VERSION_REQUIRED,
              "The compiler does not meet minimum version requirement");

using namespace dpctl::syclinterface;

/*
 * Helper function to print the metadata for a sycl::device.
 */
std::string get_device_info_str(const device &Device)
{
    std::stringstream ss;
    static constexpr const char *_endl = "\n";

    ss << std::setw(4) << " " << std::left << std::setw(16) << "Name"
       << Device.get_info<info::device::name>() << _endl << std::setw(4) << " "
       << std::left << std::setw(16) << "Driver version"
       << Device.get_info<info::device::driver_version>() << _endl
       << std::setw(4) << " " << std::left << std::setw(16) << "Vendor"
       << Device.get_info<info::device::vendor>() << _endl << std::setw(4)
       << " " << std::left << std::setw(16) << "Filter string"
       << DPCTL_GetDeviceFilterString(Device) << _endl;

    return ss.str();
}

/*!
 * @brief Canonicalizes a device identifier bit flag to have a valid (i.e., not
 * UNKNOWN) backend and device type bits.
 *
 * The device id is bit flag that indicates the backend and device type, both
 * of which are optional, that are to be queried. The function makes sure if a
 * device identifier only provides a device type value the backend is set to
 * DPCTL_ALL_BACKENDS. Similarly, if only backend is provided the device type
 * is set to DPCTL_ALL.
 *
 * @param    device_id     A bit flag storing a backend and a device type value.
 * @return   Canonicalized bit flag that makes sure neither backend nor device
 * type is UNKNOWN (0). For cases where the input device id does not provide
 * either one of the values, we set the value to ALL.
 */
int to_canonical_device_id(int device_id)
{ // If the identifier is 0 (UNKNOWN_DEVICE) return 0.
    if (!device_id)
        return 0;

    // Check if the device identifier has a backend specified. If not, then
    // toggle all backend specifier bits, i.e. set the backend to
    // DPCTL_ALL_BACKENDS.
    if (!(device_id & DPCTL_ALL_BACKENDS))
        device_id |= DPCTL_ALL_BACKENDS;

    // Check if a device type was specified. If not, set device type to ALL.
    if (!(device_id & ~DPCTL_ALL_BACKENDS))
        device_id |= DPCTL_ALL;

    return device_id;
}

struct DeviceCacheBuilder
{
    using DeviceCache = std::unordered_map<device, context>;
    /* This function implements a workaround to the current lack of a
     * default context per root device in DPC++. The map stores a "default"
     * context for each root device, and the QMgrHelper uses the map
     * whenever it creates a new queue for a root device. By doing so, we
     * avoid the performance overhead of context creation for every queue.
     *
     * The singleton pattern implemented here ensures that the map is
     * created once in a thread-safe manner. Since, the map is only read
     * post-creation we do not need any further protection to ensure
     * thread-safety.
     */
    static const DeviceCache &getDeviceCache()
    {
        static DeviceCache *cache = new DeviceCache([] {
            DeviceCache cache_l{};
            dpctl_default_selector mRanker;
            std::vector<platform> Platforms{};
            try {
                Platforms = platform::get_platforms();
            } catch (std::exception const &e) {
                error_handler(e, __FILE__, __func__, __LINE__);
                return cache_l;
            }
            for (const auto &P : Platforms) {
                auto Devices = P.get_devices();
                for (const auto &D : Devices) {
                    if (mRanker(D) < 0)
                        continue;

                    try {
                        // Per https://github.com/intel/llvm/blob/sycl/sycl/doc/
                        // extensions/supported/sycl_ext_oneapi_default_context.asciidoc
                        // sycl::queue(D) would create default platform context
                        // for capable compiler, sycl::context(D) otherwise
                        auto Q = queue(D);
                        auto Ctx = Q.get_context();
                        cache_l.emplace(D, Ctx);
                    } catch (const std::exception &e) {
                        // Nothing is added to the cache_l by guarantees of
                        // emplace
                        error_handler(e, __FILE__, __func__, __LINE__);
                    }
                }
            }
            return cache_l;
        }());

        return *cache;
    }
};

} // namespace

#undef EL
#undef EL_SYCL_TYPE
#define EL Device
#define EL_SYCL_TYPE sycl::device
#include "dpctl_vector_templ.cpp"
#undef EL
#undef EL_SYCL_TYPE

DPCTLSyclContextRef
DPCTLDeviceMgr_GetCachedContext(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTLSyclContextRef CRef = nullptr;

    auto Device = unwrap<device>(DRef);
    if (!Device) {
        error_handler("Cannot create device from DPCTLSyclDeviceRef"
                      "as input is a nullptr.",
                      __FILE__, __func__, __LINE__);
        return CRef;
    }

    using CacheT = typename DeviceCacheBuilder::DeviceCache;
    CacheT const &cache = DeviceCacheBuilder::getDeviceCache();

    if (cache.empty()) {
        // an exception was caught and logged by getDeviceCache
        return nullptr;
    }

    const auto &entry = cache.find(*Device);
    if (entry != cache.end()) {
        context *ContextPtr = nullptr;
        try {
            ContextPtr = new context(entry->second);
            CRef = wrap<context>(ContextPtr);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
            delete ContextPtr;
            CRef = nullptr;
        }
    }
    else {
        error_handler("No cached default context for device.", __FILE__,
                      __func__, __LINE__);
    }
    return CRef;
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLDeviceMgr_GetDevices(int device_identifier)
{
    using vecTy = std::vector<DPCTLSyclDeviceRef>;
    vecTy *Devices = nullptr;

    device_identifier = to_canonical_device_id(device_identifier);

    try {
        Devices = new std::vector<DPCTLSyclDeviceRef>();
    } catch (std::exception const &e) {
        delete Devices;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }

    if (!device_identifier)
        return wrap<vecTy>(Devices);

    std::vector<device> root_devices;
    try {
        root_devices = device::get_devices();
    } catch (std::exception const &e) {
        delete Devices;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
    dpctl_default_selector mRanker;

    for (const auto &root_device : root_devices) {
        if (mRanker(root_device) < 0)
            continue;
        auto Bty(DPCTL_SyclBackendToDPCTLBackendType(
            root_device.get_platform().get_backend()));
        auto Dty(DPCTL_SyclDeviceTypeToDPCTLDeviceType(
            root_device.get_info<info::device::device_type>()));
        if ((device_identifier & Bty) && (device_identifier & Dty)) {
            Devices->emplace_back(wrap<device>(new device(root_device)));
        }
    }
    // the wrap function is defined inside dpctl_vector_templ.cpp
    return wrap<vecTy>(Devices);
}

__dpctl_give const char *
DPCTLDeviceMgr_GetDeviceInfoStr(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    const char *cstr_info = nullptr;
    auto D = unwrap<device>(DRef);
    if (D) {
        try {
            auto infostr = get_device_info_str(*D);
            cstr_info = dpctl::helper::cstring_from_string(infostr);
        } catch (std::exception const &e) {
            error_handler(e, __FILE__, __func__, __LINE__);
        }
    }
    return cstr_info;
}

int DPCTLDeviceMgr_GetPositionInDevices(__dpctl_keep DPCTLSyclDeviceRef DRef,
                                        int device_identifier)
{
    static constexpr int not_found = -1;
    if (!DRef) {
        return not_found;
    }

    device_identifier = to_canonical_device_id(device_identifier);
    if (!device_identifier)
        return not_found;

    const auto &root_devices = device::get_devices();
    dpctl_default_selector mRanker;
    int index = not_found;
    const auto &reference_device = *(unwrap<device>(DRef));

    for (const auto &root_device : root_devices) {
        if (mRanker(root_device) < 0)
            continue;
        auto Bty(DPCTL_SyclBackendToDPCTLBackendType(
            root_device.get_platform().get_backend()));
        auto Dty(DPCTL_SyclDeviceTypeToDPCTLDeviceType(
            root_device.get_info<info::device::device_type>()));
        if ((device_identifier & Bty) && (device_identifier & Dty)) {
            ++index;
            if (root_device == reference_device)
                return index;
        }
    }
    return not_found;
}

/*!
 * Returns the number of available devices for a specific backend and device
 * type combination.
 */
size_t DPCTLDeviceMgr_GetNumDevices(int device_identifier)
{
    size_t nDevices = 0;
    using CacheT = typename DeviceCacheBuilder::DeviceCache;
    CacheT const &cache = DeviceCacheBuilder::getDeviceCache();

    if (cache.empty()) {
        // an exception was caught and logged by getDeviceCache
        return 0;
    }

    device_identifier = to_canonical_device_id(device_identifier);
    if (!device_identifier)
        return 0;

    dpctl_default_selector mRanker;
    for (const auto &entry : cache) {
        if (mRanker(entry.first) < 0)
            continue;
        auto Bty(DPCTL_SyclBackendToDPCTLBackendType(
            entry.first.get_platform().get_backend()));
        auto Dty(DPCTL_SyclDeviceTypeToDPCTLDeviceType(
            entry.first.get_info<info::device::device_type>()));
        if ((device_identifier & Bty) && (device_identifier & Dty))
            ++nDevices;
    }
    return nDevices;
}

/*!
 * Prints some of the device info metadata for the device corresponding to the
 * specified sycl::queue. Currently, device name, driver version, device
 * vendor, and device profile are printed out. More attributed may be added
 * later.
 */
void DPCTLDeviceMgr_PrintDeviceInfo(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto Device = unwrap<device>(DRef);
    if (Device)
        std::cout << get_device_info_str(*Device);
    else
        error_handler("Device is not valid (NULL). Cannot print device info.",
                      __FILE__, __func__, __LINE__);
}

int64_t DPCTLDeviceMgr_GetRelativeId(__dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    auto Device = unwrap<device>(DRef);

    if (Device)
        return DPCTL_GetRelativeDeviceId(*Device);

    return -1;
}

/*!
 * Returns a list of the available composite devices, or an empty list if
 * there are none.
 */
__dpctl_give DPCTLDeviceVectorRef DPCTLDeviceMgr_GetCompositeDevices()
{
    using vecTy = std::vector<DPCTLSyclDeviceRef>;
    vecTy *Devices = nullptr;

    try {
        Devices = new std::vector<DPCTLSyclDeviceRef>();
    } catch (std::exception const &e) {
        delete Devices;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }

    try {
        auto composite_devices =
            ext::oneapi::experimental::get_composite_devices();
        Devices->reserve(composite_devices.size());
        for (const auto &CDev : composite_devices) {
            Devices->emplace_back(wrap<device>(new device(std::move(CDev))));
        }
        return wrap<vecTy>(Devices);
    } catch (std::exception const &e) {
        delete Devices;
        error_handler(e, __FILE__, __func__, __LINE__);
        return nullptr;
    }
}
