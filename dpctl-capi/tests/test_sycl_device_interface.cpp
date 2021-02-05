//===--- test_sycl_device_interface.cpp - Test cases for device interface  ===//
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
/// This file has unit test cases for functions defined in
/// dpctl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//
#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "dpctl_utils.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef);

struct TestDPCTLSyclDeviceInterface
    : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    TestDPCTLSyclDeviceInterface()
    {
        DSRef = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLSyclDeviceInterface()
    {
        DPCTLDeviceSelector_Delete(DSRef);
        DPCTLDevice_Delete(DRef);
    }
};

} // namespace

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetDriverInfo)
{
    auto DriverInfo = DPCTLDevice_GetDriverInfo(DRef);
    EXPECT_TRUE(DriverInfo != nullptr);
    DPCTLCString_Delete(DriverInfo);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetMaxComputeUnits)
{
    auto n = DPCTLDevice_GetMaxComputeUnits(DRef);
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetMaxWorkItemDims)
{
    auto n = DPCTLDevice_GetMaxWorkItemDims(DRef);
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetMaxWorkItemSizes)
{
    auto item_sizes = DPCTLDevice_GetMaxWorkItemSizes(DRef);
    EXPECT_TRUE(item_sizes != nullptr);
    DPCTLSize_t_Array_Delete(item_sizes);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetMaxWorkGroupSize)
{
    auto n = DPCTLDevice_GetMaxWorkGroupSize(DRef);
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetMaxNumSubGroups)
{
    auto n = DPCTLDevice_GetMaxNumSubGroups(DRef);
    EXPECT_TRUE(n > 0);
}

// TODO: Update when DPC++ properly supports aspects
TEST_P(TestDPCTLSyclDeviceInterface, Check_HasInt64BaseAtomics)
{
    auto atomics = DPCTLDevice_HasInt64BaseAtomics(DRef);
    auto D = unwrap(DRef);
    auto has_atomics = D->has(aspect::int64_base_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

// TODO: Update when DPC++ properly supports aspects
TEST_P(TestDPCTLSyclDeviceInterface, Check_HasInt64ExtendedAtomics)
{
    auto atomics = DPCTLDevice_HasInt64ExtendedAtomics(DRef);
    auto D = unwrap(DRef);
    auto has_atomics = D->has(aspect::int64_extended_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetName)
{
    auto DevName = DPCTLDevice_GetName(DRef);
    EXPECT_TRUE(DevName != nullptr);
    DPCTLCString_Delete(DevName);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_GetVendorName)
{
    auto VendorName = DPCTLDevice_GetVendorName(DRef);
    EXPECT_TRUE(VendorName != nullptr);
    DPCTLCString_Delete(VendorName);
}

TEST_P(TestDPCTLSyclDeviceInterface, Check_DeviceTy)
{
    auto Device = unwrap(DRef);
    auto Dty = Device->get_info<info::device::device_type>();
    switch (Dty) {
    case info::device_type::accelerator:
        EXPECT_TRUE(DPCTLDevice_IsAccelerator(DRef));
        break;
    case info::device_type::cpu:
        EXPECT_TRUE(DPCTLDevice_IsCPU(DRef));
        break;
    case info::device_type::gpu:
        EXPECT_TRUE(DPCTLDevice_IsGPU(DRef));
        break;
    case info::device_type::host:
        EXPECT_TRUE(DPCTLDevice_IsHost(DRef));
        break;
    default:
        FAIL();
    }
}

INSTANTIATE_TEST_SUITE_P(TestMemberFunctions,
                         TestDPCTLSyclDeviceInterface,
                         ::testing::Values("opencl:gpu:0",
                                           "opencl:cpu:0",
                                           "level_zero:gpu:0"));
