//===-------- dpctl_sycl_device_manager.cpp - helpers for sycl devices ------=//
//
//                      Data Parallel Control (dpCtl)
//
// Copyright 2020-2021 Intel Corporation
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
#include "../helper/include/dpctl_utils_helper.h"
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_enum_types.h"
#include <CL/sycl.hpp> /* SYCL headers   */
#include <iomanip>
#include <iostream>
#include <unordered_map>

using namespace cl::sycl;

namespace
{

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(context, DPCTLSyclContextRef)

/*!
 * @brief Helper function to print the metadata for a sycl::device.
 *
 * @param    Device         My Param doc
 */
void print_device_info(const device &Device)
{
    std::stringstream ss;

    ss << std::setw(4) << " " << std::left << std::setw(16) << "Name"
       << Device.get_info<info::device::name>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Driver version"
       << Device.get_info<info::device::driver_version>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Vendor"
       << Device.get_info<info::device::vendor>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Profile"
       << Device.get_info<info::device::profile>() << '\n';
    ss << std::setw(4) << " " << std::left << std::setw(16) << "Device type";

    auto devTy = Device.get_info<info::device::device_type>();
    ss << DPCTL_DeviceTypeToStr(devTy);

    std::cout << ss.str();
}

struct DeviceWrapper
{
    device SyclDevice;
    DPCTLSyclBackendType Bty;
    DPCTLSyclDeviceType Dty;

    DeviceWrapper(const device &Device)
        : SyclDevice(Device), Bty(DPCTL_SyclBackendToDPCTLBackendType(
                                  Device.get_platform().get_backend())),
          Dty(DPCTL_SyclDeviceTypeToDPCTLDeviceType(
              Device.get_info<info::device::device_type>()))
    {
    }

    // The constructor is provided for convenience, so that we do not have to
    // lookup the BackendType and DeviceType if not needed.
    DeviceWrapper(const device &Device,
                  DPCTLSyclBackendType Bty,
                  DPCTLSyclDeviceType Dty)
        : SyclDevice(Device), Bty(Bty), Dty(Dty)
    {
    }
};

auto getHash(const device &d)
{
    if (d.is_host()) {
        return std::hash<unsigned long long>{}(-1);
    }
    else {
        return std::hash<decltype(d.get())>{}(d.get());
    }
}

struct DeviceHasher
{
    size_t operator()(const DeviceWrapper &d) const
    {
        return getHash(d.SyclDevice);
    }
};

struct DeviceEqPred
{
    bool operator()(const DeviceWrapper &d1, const DeviceWrapper &d2) const
    {
        return getHash(d1.SyclDevice) == getHash(d2.SyclDevice);
    }
};

using DeviceCache =
    std::unordered_map<DeviceWrapper, context, DeviceHasher, DeviceEqPred>;

/* This function implements a workaround to the current lack of a default
 * context per root device in DPC++. The map stores a "default" context for
 * each root device, and the QMgrHelper uses the map whenever it creates a
 * new queue for a root device. By doing so, we avoid the performance overhead
 * of context creation for every queue.
 *
 * The singleton pattern implemented here ensures that the map is created once
 * in a thread-safe manner. Since, the map is ony read post-creation we do not
 * need any further protection to ensure thread-safety.
 */
const DeviceCache &getDeviceCache()
{
    static DeviceCache cache = [] {
        DeviceCache cache_l;
        default_selector mRanker;
        auto Platforms = platform::get_platforms();
        for (const auto &P : Platforms) {
            auto Devices = P.get_devices();
            for (const auto &D : Devices) {
                if (mRanker(D) < 0)
                    continue;
                auto entry = cache_l.emplace(D, D);
                if (!entry.second) {
                    std::cerr
                        << "Fatal Error during device cache construction.\n";
                    std::terminate();
                }
            }
        }
        return cache_l;
    }();

    return cache;
}

} // namespace

#undef EL
#define EL Device
#include "dpctl_vector_templ.cpp"
#undef EL

bool DPCTLDeviceMgr_AreEq(__dpctl_keep const DPCTLSyclDeviceRef DRef1,
                          __dpctl_keep const DPCTLSyclDeviceRef DRef2)
{
    DeviceWrapper DW1(*unwrap(DRef1));
    DeviceWrapper DW2(*unwrap(DRef2));
    DeviceEqPred pred;
    return pred(DW1, DW2);
}

DPCTL_DeviceAndContextPair DPCTLDeviceMgr_GetDeviceAndContextPair(
    __dpctl_keep const DPCTLSyclDeviceRef DRef)
{
    DPCTL_DeviceAndContextPair rPair{nullptr, nullptr};
    auto Device = unwrap(DRef);
    DeviceWrapper DWrapper{*Device, DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND,
                           DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE};
    if (!Device) {
        return rPair;
    }
    auto &cache = getDeviceCache();
    auto entry = cache.find(DWrapper);
    if (entry != cache.end()) {
        try {
            rPair.DRef = wrap(new device(entry->first.SyclDevice));
            rPair.CRef = wrap(new context(entry->second));
        } catch (std::bad_alloc const &ba) {
            std::cerr << ba.what() << std::endl;
            rPair.DRef = nullptr;
            rPair.CRef = nullptr;
        }
    }
    return rPair;
}

__dpctl_give DPCTLDeviceVectorRef
DPCTLDeviceMgr_GetDevices(int device_identifier)
{
    vector_class<DPCTLSyclDeviceRef> *Devices = nullptr;
    auto &cache = getDeviceCache();

    try {
        Devices = new vector_class<DPCTLSyclDeviceRef>();
    } catch (std::bad_alloc const &ba) {
        return nullptr;
    }

    Devices->reserve(cache.size());
    for (const auto &entry : cache) {
        if ((device_identifier & entry.first.Bty) &&
            (device_identifier & entry.first.Dty))
        {
            Devices->emplace_back(wrap(new device(entry.first.SyclDevice)));
        }
    }
    // the wrap function is defined inside dpctl_vector_templ.cpp
    return wrap(Devices);
}

/*!
 * Returns the number of available devices for a specific backend and device
 * type combination.
 */
size_t DPCTLDeviceMgr_GetNumDevices(int device_identifier)
{
    size_t nDevices = 0;
    auto &cache = getDeviceCache();
    for (const auto &entry : cache)
        if ((device_identifier & entry.first.Bty) &&
            (device_identifier & entry.first.Dty))
            ++nDevices;

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
    auto Device = unwrap(DRef);
    print_device_info(*Device);
}
