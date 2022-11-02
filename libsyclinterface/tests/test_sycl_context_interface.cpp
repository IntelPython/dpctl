//===----------------- test_sycl_context_interface.cpp --------------------===//
//===---------------- Tests for  sycl context interface -------------------===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
/// dpctl_sycl_context_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_types.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <vector>

using namespace sycl;

namespace
{
} // namespace

struct TestDPCTLContextInterface : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceRef DRef = nullptr;

    TestDPCTLContextInterface()
    {
        auto DS = DPCTLFilterSelector_Create(GetParam());
        if (DS) {
            EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DS));
        }
        DPCTLDeviceSelector_Delete(DS);
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLContextInterface()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    }
};

TEST_P(TestDPCTLContextInterface, ChkCreate)
{
    DPCTLSyclContextRef CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef);
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_P(TestDPCTLContextInterface, ChkCreateWithDevices)
{
    size_t nCUs = 0;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(nCUs = DPCTLDevice_GetMaxComputeUnits(DRef));
    if (nCUs > 1) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesEqually(DRef, nCUs / 2));
        if (!DVRef) {
            GTEST_SKIP_("Skipping creating context for sub-devices");
        }
        EXPECT_NO_FATAL_FAILURE(
            CRef = DPCTLContext_CreateFromDevices(DVRef, nullptr, 0));
        ASSERT_TRUE(CRef);
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_P(TestDPCTLContextInterface, ChkCreateWithDevicesGetDevices)
{
    size_t nCUs = 0;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;
    DPCTLDeviceVectorRef Res_DVRef = nullptr;

    /* TODO: Once we have wrappers for sub-device creation let us use those
     * functions.
     */
    EXPECT_NO_FATAL_FAILURE(nCUs = DPCTLDevice_GetMaxComputeUnits(DRef));
    if (nCUs > 1) {
        EXPECT_NO_FATAL_FAILURE(
            DVRef = DPCTLDevice_CreateSubDevicesEqually(DRef, nCUs / 2));
        if (!DVRef) {
            GTEST_SKIP_("Skipping creating context for sub-devices");
        }
        EXPECT_NO_FATAL_FAILURE(
            CRef = DPCTLContext_CreateFromDevices(DVRef, nullptr, 0));
        ASSERT_TRUE(CRef);
        const size_t len = DPCTLDeviceVector_Size(DVRef);
        ASSERT_TRUE(DPCTLContext_DeviceCount(CRef) == len);
        EXPECT_NO_FATAL_FAILURE(Res_DVRef = DPCTLContext_GetDevices(CRef));
        ASSERT_TRUE(DPCTLDeviceVector_Size(Res_DVRef) == len);
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(Res_DVRef));
}

TEST_P(TestDPCTLContextInterface, ChkGetDevices)
{
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef);
    EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLContext_GetDevices(CRef));
    ASSERT_TRUE(DVRef);
    EXPECT_TRUE(DPCTLDeviceVector_Size(DVRef) == 1);
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
}

TEST_P(TestDPCTLContextInterface, ChkAreEq)
{
    DPCTLSyclContextRef CRef1 = nullptr, CRef2 = nullptr, CRef3 = nullptr;
    bool are_eq = true, are_not_eq = false;

    EXPECT_NO_FATAL_FAILURE(CRef1 = DPCTLContext_Create(DRef, nullptr, 0));
    EXPECT_NO_FATAL_FAILURE(CRef2 = DPCTLContext_Copy(CRef1));
    // TODO: This work till DPC++ does not have a default context per device,
    // after that we need to change the test case some how.
    EXPECT_NO_FATAL_FAILURE(CRef3 = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef3);
    ASSERT_TRUE(CRef2);
    ASSERT_TRUE(CRef1);

    EXPECT_NO_FATAL_FAILURE(are_eq = DPCTLContext_AreEq(CRef1, CRef2));
    EXPECT_NO_FATAL_FAILURE(are_not_eq = DPCTLContext_AreEq(CRef1, CRef3));
    EXPECT_TRUE(are_eq);
    EXPECT_FALSE(are_not_eq);
    EXPECT_TRUE(DPCTLContext_Hash(CRef1) == DPCTLContext_Hash(CRef2));
    EXPECT_FALSE(DPCTLContext_Hash(CRef1) == DPCTLContext_Hash(CRef3));

    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef1));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef2));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef3));
}

TEST_P(TestDPCTLContextInterface, ChkIsHost)
{
    DPCTLSyclContextRef CRef = nullptr;
    bool is_host_device = false, is_host_context = false;

    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef);

    EXPECT_NO_FATAL_FAILURE(is_host_device = DPCTLDevice_IsHost(DRef));
    EXPECT_NO_FATAL_FAILURE(is_host_context = DPCTLContext_IsHost(CRef));
    EXPECT_TRUE(is_host_device == is_host_context);

    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_P(TestDPCTLContextInterface, ChkGetBackend)
{
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclBackendType context_backend = DPCTL_UNKNOWN_BACKEND,
                         device_backend = DPCTL_UNKNOWN_BACKEND;

    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef);

    EXPECT_NO_FATAL_FAILURE(device_backend = DPCTLDevice_GetBackend(DRef));
    EXPECT_NO_FATAL_FAILURE(context_backend = DPCTLContext_GetBackend(CRef));
    EXPECT_TRUE(device_backend == context_backend);

    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

INSTANTIATE_TEST_SUITE_P(DPCTLContextTests,
                         TestDPCTLContextInterface,
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

struct TestDPCTLContextNullArgs : public ::testing::Test
{
    DPCTLSyclContextRef Null_CRef = nullptr;
    DPCTLSyclDeviceRef Null_DRef = nullptr;
    DPCTLDeviceVectorRef Null_DVRef = nullptr;
    TestDPCTLContextNullArgs() = default;
    ~TestDPCTLContextNullArgs() = default;
};

TEST_F(TestDPCTLContextNullArgs, ChkCreate)
{
    DPCTLSyclContextRef CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(Null_DRef, nullptr, 0));
    ASSERT_FALSE(bool(CRef));
}

TEST_F(TestDPCTLContextNullArgs, ChkCreateFromDevices)
{
    DPCTLSyclContextRef CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(
        CRef = DPCTLContext_CreateFromDevices(Null_DVRef, nullptr, 0));
    ASSERT_FALSE(bool(CRef));
}

TEST_F(TestDPCTLContextNullArgs, ChkAreEq)
{
    DPCTLSyclContextRef Null_C2Ref = nullptr;
    bool are_eq = true;
    EXPECT_NO_FATAL_FAILURE(are_eq = DPCTLContext_AreEq(Null_CRef, Null_C2Ref));
    ASSERT_FALSE(are_eq);
}

TEST_F(TestDPCTLContextNullArgs, ChkCopy)
{
    DPCTLSyclContextRef Copied_CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_CRef = DPCTLContext_Copy(Null_CRef));
    ASSERT_FALSE(bool(Copied_CRef));
}

TEST_F(TestDPCTLContextNullArgs, ChkGetDevices)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLContext_GetDevices(Null_CRef));
    ASSERT_FALSE(bool(DVRef));
}

TEST_F(TestDPCTLContextNullArgs, ChkDeviceCount)
{
    size_t count = -1;
    EXPECT_NO_FATAL_FAILURE(count = DPCTLContext_DeviceCount(Null_CRef));
    ASSERT_TRUE(count == 0);
}

TEST_F(TestDPCTLContextNullArgs, ChkIsHost)
{
    bool is_host = true;
    EXPECT_NO_FATAL_FAILURE(is_host = DPCTLContext_IsHost(Null_CRef));
    ASSERT_FALSE(is_host);
}

TEST_F(TestDPCTLContextNullArgs, ChkHash)
{
    size_t hash = 0;
    EXPECT_NO_FATAL_FAILURE(hash = DPCTLContext_Hash(Null_CRef));
    ASSERT_TRUE(hash == 0);
}

TEST_F(TestDPCTLContextNullArgs, ChkGetBackend)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    EXPECT_NO_FATAL_FAILURE(BTy = DPCTLContext_GetBackend(Null_CRef));
    ASSERT_TRUE(BTy == DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND);
}

TEST_F(TestDPCTLContextNullArgs, ChkDelete)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(Null_CRef));
}
