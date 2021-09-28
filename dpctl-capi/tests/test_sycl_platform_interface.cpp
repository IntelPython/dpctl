//===-- test_sycl_platform_interface.cpp - Test cases for platform interface =//
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
/// This file has unit test cases for functions defined in
/// dpctl_sycl_platform_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_sycl_platform_manager.h"
#include "dpctl_utils.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <vector>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::vector<DPCTLSyclPlatformRef>,
                                   DPCTLPlatformVectorRef);

void check_platform_name(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    const char *name = nullptr;
    EXPECT_NO_FATAL_FAILURE(name = DPCTLPlatform_GetName(PRef));
    EXPECT_TRUE(name);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(name));
}

void check_platform_vendor(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    const char *vname = nullptr;
    EXPECT_NO_FATAL_FAILURE(vname = DPCTLPlatform_GetVendor(PRef));
    EXPECT_TRUE(vname);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(vname));
}

void check_platform_version(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    const char *version = nullptr;
    EXPECT_NO_FATAL_FAILURE(version = DPCTLPlatform_GetVersion(PRef));
    EXPECT_TRUE(version);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(version));
}

void check_platform_backend(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    EXPECT_NO_FATAL_FAILURE(BTy = DPCTLPlatform_GetBackend(PRef));
    EXPECT_TRUE([BTy] {
        switch (BTy) {
        case DPCTLSyclBackendType::DPCTL_CUDA:
            return true;
        case DPCTLSyclBackendType::DPCTL_HOST:
            return true;
        case DPCTLSyclBackendType::DPCTL_LEVEL_ZERO:
            return true;
        case DPCTLSyclBackendType::DPCTL_OPENCL:
            return true;
        default:
            return false;
        }
    }());
}

} // namespace

struct TestDPCTLSyclPlatformInterface
    : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclPlatformRef PRef = nullptr;

    TestDPCTLSyclPlatformInterface()
    {
        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLFilterSelector_Create(GetParam()));
        if (DSRef) {
            EXPECT_NO_FATAL_FAILURE(
                PRef = DPCTLPlatform_CreateFromSelector(DSRef));
        }
    }

    void SetUp()
    {
        if (!PRef) {
            auto message = "Skipping as no platform of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLSyclPlatformInterface()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(PRef));
    }
};

struct TestDPCTLSyclDefaultPlatform : public ::testing::Test
{
    DPCTLSyclPlatformRef PRef = nullptr;

    TestDPCTLSyclDefaultPlatform()
    {
        EXPECT_NO_FATAL_FAILURE(PRef = DPCTLPlatform_Create());
    }

    void SetUp()
    {
        ASSERT_TRUE(PRef);
    }

    ~TestDPCTLSyclDefaultPlatform()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(PRef));
    }
};

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetName)
{
    check_platform_name(PRef);
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetVendor)
{
    check_platform_vendor(PRef);
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetVersion)
{
    check_platform_version(PRef);
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetBackend)
{
    check_platform_backend(PRef);
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkCopy)
{
    DPCTLSyclPlatformRef Copied_PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_PRef = DPCTLPlatform_Copy(PRef));
    EXPECT_TRUE(bool(Copied_PRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(Copied_PRef));
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkPrintInfo)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformMgr_PrintInfo(PRef, 0));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetName)
{
    check_platform_name(PRef);
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetVendor)
{
    check_platform_vendor(PRef);
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetVersion)
{
    check_platform_version(PRef);
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetBackend)
{
    check_platform_backend(PRef);
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkPrintInfo0)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformMgr_PrintInfo(PRef, 0));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkPrintInfo1)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformMgr_PrintInfo(PRef, 1));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkPrintInfo2)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformMgr_PrintInfo(PRef, 2));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkPrintInfo3)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformMgr_PrintInfo(PRef, 3));
}

TEST(TestGetPlatforms, Chk)
{
    auto PVRef = DPCTLPlatform_GetPlatforms();
    auto nPlatforms = DPCTLPlatformVector_Size(PVRef);
    if (nPlatforms) {
        for (auto i = 0ul; i < nPlatforms; ++i) {
            DPCTLSyclPlatformRef PRef = nullptr;
            EXPECT_NO_FATAL_FAILURE(PRef = DPCTLPlatformVector_GetAt(PVRef, i));
            ASSERT_TRUE(PRef);
            check_platform_backend(PRef);
            check_platform_name(PRef);
            check_platform_vendor(PRef);
            check_platform_version(PRef);
            EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(PRef));
        }
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformVector_Delete(PVRef));
}

INSTANTIATE_TEST_SUITE_P(DPCTLPlatformTests,
                         TestDPCTLSyclPlatformInterface,
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
