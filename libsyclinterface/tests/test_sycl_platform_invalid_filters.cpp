//=== test_sycl_platform_invalid_filters.cpp - -ve tests for platform iface ==//
//
//                      Data Parallel Control (dpctl)
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
/// Negative test cases for SYCL platform creation with invalid filter
/// selectors.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_platform_interface.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;
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

TEST_P(TestUnsupportedFilters, ChkDPCTLPlatformCreateFromSelector)
{
    DPCTLSyclPlatformRef PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(PRef = DPCTLPlatform_CreateFromSelector(DSRef));
    ASSERT_TRUE(PRef == nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(PRef));
}

INSTANTIATE_TEST_SUITE_P(
    NegativeDeviceCreationTests,
    TestUnsupportedFilters,
    ::testing::Values("abc", "-1", "invalid_filter", "cuda:cpu:0"));
