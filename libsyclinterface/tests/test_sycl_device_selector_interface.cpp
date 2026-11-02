//===--- test_sycl_device_selector_interface.cpp - Device selector tests   ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2021 Intel Corporation
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
#include "dpctl_utils_helper.h"
#include <gtest/gtest.h>
#include <string>
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

TEST_F(TestDeviceSelectorInterface, ChkDPCTLFilterSelectorScore)
{
    using dpctl::syclinterface::unwrap;
    using dpctl::syclinterface::wrap;

    const auto devices = sycl::device::get_devices();
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const std::string filters[] = {std::to_string(i),
                                       DPCTL_GetDeviceFilterString(devices[i])};
        for (const auto &fs : filters) {
            DPCTLSyclDeviceSelectorRef DSRef = nullptr;
            EXPECT_NO_FATAL_FAILURE(DSRef =
                                        DPCTLFilterSelector_Create(fs.c_str()));
            ASSERT_TRUE(DSRef != nullptr);
            for (std::size_t j = 0; j < devices.size(); ++j) {
                auto D = devices[j];
                auto DRef = wrap<sycl::device>(&D);
                int score = DPCTLDeviceSelector_Score(DSRef, DRef);
                if (i == j)
                    EXPECT_TRUE(score >= 0) << fs;
                else
                    EXPECT_TRUE(score < 0) << fs;
            }
            EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        }
    }
}

TEST_F(TestDeviceSelectorInterface, ChkDPCTLFilterSelectorList)
{
    const auto devices = sycl::device::get_devices();
    if (devices.size() < 2)
        GTEST_SKIP_("Need at least two devices.");

    // "0,1" must accept both device 0 and device 1
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create("0,1"));
    ASSERT_TRUE(DSRef != nullptr);
    for (std::size_t j = 0; j < devices.size(); ++j) {
        auto D = devices[j];
        auto DRef = dpctl::syclinterface::wrap<sycl::device>(&D);
        int score = DPCTLDeviceSelector_Score(DSRef, DRef);
        if (j < 2)
            EXPECT_TRUE(score >= 0);
        else
            EXPECT_TRUE(score < 0);
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
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
                                           "0",
                                           "gpu:opencl",
                                           "0:gpu",
                                           "opencl::0",
                                           ":",
                                           "gpu,cpu",
                                           "cuda,cpu"));

INSTANTIATE_TEST_SUITE_P(NegativeFilterSelectorCreation,
                         TestUnsupportedFilters,
                         ::testing::Values("abc",
                                           "-1",
                                           "cuda:cpu:0",
                                           "",
                                           ",",
                                           "OPENCL",
                                           " opencl",
                                           "opencl:gpu:cpu",
                                           "opencl:level_zero",
                                           "0:1",
                                           "opencl:gpu:0:0",
                                           "99999999999"));
