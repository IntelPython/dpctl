//===------- test_sycl_device_manager.cpp - Test cases for device manager  ===//
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
/// dpctl_sycl_device_manager.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_device_selector_interface.h"
#include <gtest/gtest.h>

struct TestDPCTLDeviceManager : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    TestDPCTLDeviceManager()
    {
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(GetParam()));
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLDeviceManager()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
};

TEST_P(TestDPCTLDeviceManager, Chk_AreEq)
{
    auto DRef2 = DPCTLDevice_CreateFromSelector(DSRef);
    bool compare = false;
    EXPECT_NO_FATAL_FAILURE(compare = DPCTLDeviceMgr_AreEq(DRef, DRef2));
    EXPECT_TRUE(compare);
}

TEST_P(TestDPCTLDeviceManager, Chk_PrintDeviceInfo)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
}

TEST_P(TestDPCTLDeviceManager, Chk_GetDeviceAndContextPair)
{
    DPCTL_DeviceAndContextPair deviceAndContext;
    EXPECT_NO_FATAL_FAILURE(deviceAndContext =
                                DPCTLDeviceMgr_GetDeviceAndContextPair(DRef));
    ASSERT_TRUE(deviceAndContext.CRef != nullptr);
    ASSERT_TRUE(deviceAndContext.DRef != nullptr);
}

INSTANTIATE_TEST_SUITE_P(DeviceMgrFunctions,
                         TestDPCTLDeviceManager,
                         ::testing::Values("opencl:gpu:0",
                                           "opencl:cpu:0",
                                           "level_zero:gpu:0"));

struct TestDPCTLDeviceVector : public ::testing::TestWithParam<int>
{
    DPCTLDeviceVectorRef DV = nullptr;
    size_t nDevices = 0;

    TestDPCTLDeviceVector()
    {
        EXPECT_NO_FATAL_FAILURE(DV = DPCTLDeviceMgr_GetDevices(GetParam()));
        EXPECT_TRUE(DV != nullptr);
        EXPECT_NO_FATAL_FAILURE(nDevices = DPCTLDeviceVector_Size(DV));
    }

    void SetUp()
    {
        if (!nDevices) {
            GTEST_SKIP_("Skipping as no devices returned for identifier");
        }
    }

    ~TestDPCTLDeviceVector()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Clear(DV));
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DV));
    }
};

TEST_P(TestDPCTLDeviceVector, Chk_GetAt)
{
    for (auto i = 0ul; i < nDevices; ++i) {
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDeviceVector_GetAt(DV, i));
        ASSERT_TRUE(DRef != nullptr);
    }
}

INSTANTIATE_TEST_SUITE_P(
    GetDevices,
    TestDPCTLDeviceVector,
    ::testing::Values(DPCTLSyclBackendType::DPCTL_HOST,
                      DPCTLSyclBackendType::DPCTL_LEVEL_ZERO,
                      DPCTLSyclBackendType::DPCTL_OPENCL,
                      DPCTLSyclBackendType::DPCTL_OPENCL |
                          DPCTLSyclDeviceType::DPCTL_GPU));

TEST(TestDPCTLDeviceVector, Chk_DPCTLDeviceVector_Create)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    size_t nDevices = 0;
    EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLDeviceVector_Create());
    ASSERT_TRUE(DVRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(nDevices = DPCTLDeviceVector_Size(DVRef));
    EXPECT_TRUE(nDevices == 0);
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
}
