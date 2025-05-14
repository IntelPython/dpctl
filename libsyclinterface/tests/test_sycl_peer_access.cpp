//===--- test_sycl_peer_access.cpp - Test cases for device peer access  ===//
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
/// This file has unit test cases for peer access functions defined in
/// dpctl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_utils_helper.h"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

struct TestDPCTLPeerAccess : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclPlatformRef P = nullptr;
    DPCTLDeviceVectorRef DV = nullptr;

    TestDPCTLPeerAccess()
    {
        auto DS = DPCTLFilterSelector_Create(GetParam());
        if (DS) {
            EXPECT_NO_FATAL_FAILURE(P = DPCTLPlatform_CreateFromSelector(DS));
        }
        DPCTLDeviceSelector_Delete(DS);
        if (P) {
            DV = DPCTLPlatform_GetDevices(P, DPCTLSyclDeviceType::DPCTL_ALL);
        }
    }

    void SetUp()
    {
        if (!P || !DV) {
            auto message = "Skipping as no devices of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }

        if (DPCTLDeviceVector_Size(DV) < 2) {
            GTEST_SKIP_("Peer access tests require more than one device.");
        }
    }

    ~TestDPCTLPeerAccess()
    {
        DPCTLDeviceVector_Delete(DV);
        DPCTLPlatform_Delete(P);
    }
};

TEST_P(TestDPCTLPeerAccess, ChkAccessSupported)
{
    auto D0 = DPCTLDeviceVector_GetAt(DV, 0);
    auto D1 = DPCTLDeviceVector_GetAt(DV, 1);
    ASSERT_TRUE(D0 != nullptr);
    ASSERT_TRUE(D1 != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_CanAccessPeer(
        D0, D1, DPCTLPeerAccessType::access_supported));
}

TEST_P(TestDPCTLPeerAccess, ChkAtomicsSupported)
{
    auto D0 = DPCTLDeviceVector_GetAt(DV, 0);
    auto D1 = DPCTLDeviceVector_GetAt(DV, 1);
    ASSERT_TRUE(D0 != nullptr);
    ASSERT_TRUE(D1 != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_CanAccessPeer(
        D0, D1, DPCTLPeerAccessType::atomics_supported));
}

TEST_P(TestDPCTLPeerAccess, ChkPeerAccess)
{
    auto D0 = DPCTLDeviceVector_GetAt(DV, 0);
    auto D1 = DPCTLDeviceVector_GetAt(DV, 1);
    ASSERT_TRUE(D0 != nullptr);
    ASSERT_TRUE(D1 != nullptr);
    bool canEnable = false;
    EXPECT_NO_FATAL_FAILURE(canEnable = DPCTLDevice_CanAccessPeer(
                                D0, D1, DPCTLPeerAccessType::access_supported));
    if (canEnable) {
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_EnablePeerAccess(D0, D1));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_DisablePeerAccess(D0, D1));
    }
}

TEST_P(TestDPCTLPeerAccess, ChkPeerAccessToSelf)
{
    auto D0 = DPCTLDeviceVector_GetAt(DV, 0);
    ASSERT_TRUE(D0 != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_CanAccessPeer(
        D0, D0, DPCTLPeerAccessType::access_supported));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_EnablePeerAccess(D0, D0));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_DisablePeerAccess(D0, D0));
}

INSTANTIATE_TEST_SUITE_P(DPCTLDeviceFns,
                         TestDPCTLPeerAccess,
                         ::testing::Values("level_zero", "cuda", "hip"));

struct TestDPCTLPeerAccessNullArgs : public ::testing::Test
{
    DPCTLSyclDeviceRef Null_DR0 = nullptr;
    DPCTLSyclDeviceRef Null_DR1 = nullptr;
};

TEST_F(TestDPCTLPeerAccessNullArgs, ChkAccessSupported)
{
    bool accessSupported = true;
    EXPECT_NO_FATAL_FAILURE(
        accessSupported = DPCTLDevice_CanAccessPeer(
            Null_DR0, Null_DR1, DPCTLPeerAccessType::access_supported));
    ASSERT_FALSE(accessSupported);
}

TEST_F(TestDPCTLPeerAccessNullArgs, ChkAtomicsSupported)
{
    bool accessSupported = true;
    EXPECT_NO_FATAL_FAILURE(
        accessSupported = DPCTLDevice_CanAccessPeer(
            Null_DR0, Null_DR1, DPCTLPeerAccessType::atomics_supported));
    ASSERT_FALSE(accessSupported);
}

TEST_F(TestDPCTLPeerAccessNullArgs, ChkPeerAccess)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_EnablePeerAccess(Null_DR0, Null_DR1));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_DisablePeerAccess(Null_DR0, Null_DR1));
}
