//===--- test_sycl_device_selector_interface.cpp - Device selector tests   ===//
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
/// This file has unit test cases for functions defined in
/// dpctl_sycl_device_selector_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_type_casters.hpp"
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

using namespace sycl;

struct TestDeviceSelectorInterface : public ::testing::Test
{
};

struct TestFilterSelector : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    TestFilterSelector()
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

    ~TestFilterSelector()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
};

struct TestUnsupportedFilters : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;

    TestUnsupportedFilters()
    {
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(GetParam()));
    }

    ~TestUnsupportedFilters()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
};

TEST_F(TestDeviceSelectorInterface, ChkDPCTLAcceleratorSelectorCreate)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLAcceleratorSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));

    if (!DRef) {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
        GTEST_SKIP_("Device not found. Skip tests.");
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
    EXPECT_TRUE(DPCTLDevice_IsAccelerator(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_F(TestDeviceSelectorInterface, ChkDPCTLDefaultSelectorCreate)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));

    if (!DRef) {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
        GTEST_SKIP_("Device not found. Skip tests.");
    }

    ASSERT_TRUE(DRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_F(TestDeviceSelectorInterface, ChkDPCTLCPUSelectorCreate)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLCPUSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));

    if (!DRef) {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
        GTEST_SKIP_("Device not found. Skip tests.");
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
    EXPECT_TRUE(DPCTLDevice_IsCPU(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_F(TestDeviceSelectorInterface, ChkDPCTLGPUSelectorCreate)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLGPUSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));

    if (!DRef) {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
        GTEST_SKIP_("Device not found. Skip tests.");
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceMgr_PrintDeviceInfo(DRef));
    EXPECT_TRUE(DPCTLDevice_IsGPU(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_P(TestFilterSelector, ChkDPCTLFilterSelectorCreate)
{
    ASSERT_TRUE(DRef != nullptr);
}

TEST_P(TestUnsupportedFilters, ChkDPCTLFilterSelectorCreate)
{
    DPCTLSyclDeviceRef DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    ASSERT_TRUE(DRef == nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
}

TEST_F(TestDeviceSelectorInterface, ChkDPCTLGPUSelectorScore)
{
    DPCTLSyclDeviceSelectorRef DSRef_GPU = nullptr;
    DPCTLSyclDeviceSelectorRef DSRef_CPU = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef_GPU = DPCTLGPUSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DSRef_CPU = DPCTLCPUSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef_CPU));

    if (!DRef) {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef_GPU));
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef_CPU));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
        GTEST_SKIP_("Device not found. Skip tests.");
    }

    ASSERT_TRUE(DRef != nullptr);
    EXPECT_TRUE(DPCTLDevice_IsCPU(DRef));
    EXPECT_TRUE(DPCTLDeviceSelector_Score(DSRef_GPU, DRef) < 0);
    EXPECT_TRUE(DPCTLDeviceSelector_Score(DSRef_CPU, DRef) > 0);
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef_GPU));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef_CPU));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));

    DPCTLSyclDeviceSelectorRef Null_DSRef = nullptr;
    DPCTLSyclDeviceRef Null_DRef = nullptr;
    int score = 1;
    EXPECT_NO_FATAL_FAILURE(
        score = DPCTLDeviceSelector_Score(Null_DSRef, Null_DRef));
    ASSERT_TRUE(score < 0);
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
                                           "1",
                                           "0"));

INSTANTIATE_TEST_SUITE_P(NegativeFilterSelectorCreation,
                         TestUnsupportedFilters,
                         ::testing::Values("abc", "-1", "cuda:cpu:0"));
