//===--- test_sycl_device_selector_interface.cpp - Device selector tests   ===//
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
/// dpctl_sycl_device_selector_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_device_selector_interface.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device, DPCTLSyclDeviceRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(device_selector, DPCTLSyclDeviceSelectorRef)
} // namespace

struct TestDeviceSelectorInterface : public ::testing::Test
{
};

struct TestFilterSelector : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;

    TestFilterSelector()
    {
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(GetParam()));
    }

    void SetUp()
    {
        if (!DSRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestFilterSelector()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
};

struct TestUnsupportedFilters : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;

    TestUnsupportedFilters()
    {
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(GetParam()));
    }

    void SetUp()
    {
        if (!DSRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestUnsupportedFilters()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
};

TEST_F(TestDeviceSelectorInterface, Chk_DPCTLAcceleratorSelector_Create)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLAcceleratorSelector_Create());
    if (DSRef) {
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        ASSERT_TRUE(DRef != nullptr);
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
        EXPECT_TRUE(DPCTLDevice_IsAccelerator(DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST_F(TestDeviceSelectorInterface, Chk_DPCTLDefaultSelector_Create)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    if (DSRef) {
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        ASSERT_TRUE(DRef != nullptr);
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST_F(TestDeviceSelectorInterface, Chk_DPCTLCPUSelector_Create)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLCPUSelector_Create());
    if (DSRef) {
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        ASSERT_TRUE(DRef != nullptr);
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
        EXPECT_TRUE(DPCTLDevice_IsCPU(DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST_F(TestDeviceSelectorInterface, Chk_DPCTLGPUSelector_Create)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLGPUSelector_Create());
    if (DSRef) {
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        ASSERT_TRUE(DRef != nullptr);
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
        EXPECT_TRUE(DPCTLDevice_IsGPU(DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST_F(TestDeviceSelectorInterface, Chk_DPCTLHostSelector_Create)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLHostSelector_Create());
    if (DSRef) {
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        ASSERT_TRUE(DRef != nullptr);
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
        // FIXME: DPCPP's host_selector returns a CPU device for some reason.
        // EXPECT_TRUE(DPCTLDevice_IsHost(DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
}

TEST_P(TestFilterSelector, Chk_DPCTLFilterSelector_Create)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    ASSERT_TRUE(DRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestUnsupportedFilters, Chk_DPCTLFilterSelector_Create)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    ASSERT_TRUE(DRef == nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

INSTANTIATE_TEST_SUITE_P(FilterSelectorCreation,
                         TestFilterSelector,
                         ::testing::Values("opencl",
                                           "opencl:gpu",
                                           "opencl:cpu",
                                           "opencl:gpu:0",
                                           "gpu",
                                           "cpu",
                                           "level_zero",
                                           "level_zero:gpu",
                                           "opencl:cpu:0",
                                           "level_zero:gpu:0",
                                           "gpu:0",
                                           "gpu:1",
                                           "1"));

INSTANTIATE_TEST_SUITE_P(
    NegativeFilterSelectorCreation,
    TestUnsupportedFilters,
    ::testing::Values("abc", "-1", "opencl:gpu:1", "level_zero:cpu:0"));
