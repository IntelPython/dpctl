//===-- test_sycl_platform_interface.cpp - Test cases for platform interface =//
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
/// dpctl_sycl_platform_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_sycl_platform_manager.h"
#include "dpctl_utils.h"
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

namespace
{

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
        case DPCTLSyclBackendType::DPCTL_HIP:
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

void check_platform_default_context(
    __dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    DPCTLSyclContextRef CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLPlatform_GetDefaultContext(PRef));
    EXPECT_TRUE(CRef != nullptr);

    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

void check_platform_get_devices(__dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    size_t nDevices = 0;

    DPCTLSyclDeviceType defDTy = DPCTLSyclDeviceType::DPCTL_ALL;
    EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLPlatform_GetDevices(PRef, defDTy));
    EXPECT_TRUE(DVRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(nDevices = DPCTLDeviceVector_Size(DVRef));
    for (auto i = 0ul; i < nDevices; ++i) {
        DPCTLSyclDeviceRef DRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDeviceVector_GetAt(DVRef, i));
        ASSERT_TRUE(DRef != nullptr);
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Clear(DVRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
}

void check_platform_get_composite_devices(
    __dpctl_keep const DPCTLSyclPlatformRef PRef)
{
    DPCTLDeviceVectorRef CDVRef = nullptr;
    size_t nCDevices = 0;

    EXPECT_NO_FATAL_FAILURE(CDVRef = DPCTLPlatform_GetCompositeDevices(PRef));
    EXPECT_TRUE(CDVRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(nCDevices = DPCTLDeviceVector_Size(CDVRef));
    for (auto i = 0ul; i < nCDevices; ++i) {
        DPCTLSyclDeviceRef CDRef = nullptr;
        EXPECT_NO_FATAL_FAILURE(CDRef = DPCTLDeviceVector_GetAt(CDVRef, i));
        ASSERT_TRUE(CDRef != nullptr);
        ASSERT_TRUE(
            DPCTLDevice_HasAspect(CDRef, DPCTLSyclAspectType::is_composite));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(CDRef));
    }

    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Clear(CDVRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(CDVRef));
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

struct TestDPCTLSyclPlatformNull : public ::testing::Test
{
    DPCTLSyclPlatformRef NullPRef = nullptr;
    DPCTLSyclDeviceSelectorRef NullDSRef = nullptr;

    TestDPCTLSyclPlatformNull() = default;
    ~TestDPCTLSyclPlatformNull() = default;
};

TEST_F(TestDPCTLSyclPlatformNull, ChkCopy)
{
    DPCTLSyclPlatformRef Copied_PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_PRef = DPCTLPlatform_Copy(NullPRef));
    ASSERT_TRUE(Copied_PRef == nullptr);
}

TEST_F(TestDPCTLSyclPlatformNull, ChkCreateFromSelector)
{
    DPCTLSyclPlatformRef Created_PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Created_PRef =
                                DPCTLPlatform_CreateFromSelector(NullDSRef));
    ASSERT_TRUE(Created_PRef == nullptr);
}

TEST_F(TestDPCTLSyclPlatformNull, ChkGetBackend)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    EXPECT_NO_FATAL_FAILURE(BTy = DPCTLPlatform_GetBackend(NullPRef));
    ASSERT_TRUE(BTy == DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND);
}

TEST_F(TestDPCTLSyclPlatformNull, ChkGetName)
{
    const char *name = nullptr;
    EXPECT_NO_FATAL_FAILURE(name = DPCTLPlatform_GetName(NullPRef));
    ASSERT_TRUE(name == nullptr);
}

TEST_F(TestDPCTLSyclPlatformNull, ChkGetVendor)
{
    const char *vendor = nullptr;
    EXPECT_NO_FATAL_FAILURE(vendor = DPCTLPlatform_GetVendor(NullPRef));
    ASSERT_TRUE(vendor == nullptr);
}

TEST_F(TestDPCTLSyclPlatformNull, ChkGetVersion)
{
    const char *version = nullptr;
    EXPECT_NO_FATAL_FAILURE(version = DPCTLPlatform_GetVersion(NullPRef));
    ASSERT_TRUE(version == nullptr);
}

TEST_F(TestDPCTLSyclPlatformNull, ChkGetDefaultConext)
{
    DPCTLSyclContextRef CRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLPlatform_GetDefaultContext(NullPRef));
    EXPECT_TRUE(CRef == nullptr);
}

struct TestDPCTLSyclDefaultPlatform : public ::testing::Test
{
    DPCTLSyclPlatformRef PRef = nullptr;

    TestDPCTLSyclDefaultPlatform()
    {
        EXPECT_NO_FATAL_FAILURE(PRef = DPCTLPlatform_Create());
    }

    void SetUp() { ASSERT_TRUE(PRef); }

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

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetDefaultContext)
{
    check_platform_default_context(PRef);
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkCopy)
{
    DPCTLSyclPlatformRef Copied_PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_PRef = DPCTLPlatform_Copy(PRef));
    EXPECT_TRUE(bool(Copied_PRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(Copied_PRef));
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkCopyNullArg)
{
    DPCTLSyclPlatformRef Null_PRef = nullptr;
    DPCTLSyclPlatformRef Copied_PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_PRef = DPCTLPlatform_Copy(Null_PRef));
    EXPECT_FALSE(bool(Copied_PRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(Copied_PRef));
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetInfo)
{
    const char *info_str = nullptr;
    EXPECT_NO_FATAL_FAILURE(info_str = DPCTLPlatformMgr_GetInfo(PRef, 0));
    ASSERT_TRUE(info_str != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(info_str));
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkPrintInfo)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformMgr_PrintInfo(PRef, 0));
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkPrintInfoNullArg)
{
    DPCTLSyclPlatformRef Null_PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatformMgr_PrintInfo(Null_PRef, 0));
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkAreEq)
{
    DPCTLSyclPlatformRef PRef_Copy = nullptr;

    EXPECT_NO_FATAL_FAILURE(PRef_Copy = DPCTLPlatform_Copy(PRef));

    ASSERT_TRUE(DPCTLPlatform_AreEq(PRef, PRef_Copy));
    EXPECT_TRUE(DPCTLPlatform_Hash(PRef) == DPCTLPlatform_Hash(PRef_Copy));

    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(PRef_Copy));
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkAreEqNullArg)
{
    DPCTLSyclPlatformRef Null_PRef = nullptr;
    ASSERT_FALSE(DPCTLPlatform_AreEq(PRef, Null_PRef));
    ASSERT_TRUE(DPCTLPlatform_Hash(Null_PRef) == 0);
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetDevices)
{
    check_platform_get_devices(PRef);
}

TEST_P(TestDPCTLSyclPlatformInterface, ChkGetCompositeDevices)
{
    check_platform_get_composite_devices(PRef);
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetName) { check_platform_name(PRef); }

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

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetInfo0)
{
    const char *info_str = nullptr;
    EXPECT_NO_FATAL_FAILURE(info_str = DPCTLPlatformMgr_GetInfo(PRef, 0));
    ASSERT_TRUE(info_str != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(info_str));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetInfo1)
{
    const char *info_str = nullptr;
    EXPECT_NO_FATAL_FAILURE(info_str = DPCTLPlatformMgr_GetInfo(PRef, 1));
    ASSERT_TRUE(info_str != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(info_str));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetInfo2)
{
    const char *info_str = nullptr;
    EXPECT_NO_FATAL_FAILURE(info_str = DPCTLPlatformMgr_GetInfo(PRef, 2));
    ASSERT_TRUE(info_str != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(info_str));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetInfo3)
{
    const char *info_str = nullptr;
    EXPECT_NO_FATAL_FAILURE(info_str = DPCTLPlatformMgr_GetInfo(PRef, 3));
    ASSERT_TRUE(info_str != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(info_str));
}

TEST_F(TestDPCTLSyclDefaultPlatform, ChkGetInfoNull)
{
    const char *info_str = nullptr;
    DPCTLSyclPlatformRef NullPRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(info_str = DPCTLPlatformMgr_GetInfo(NullPRef, 0));
    ASSERT_TRUE(info_str == nullptr);
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
