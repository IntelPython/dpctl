//===--- test_helper.cpp - Test cases for helper functions  ===//
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
/// helper/include/dpctl_utils_helper.h.
///
//===----------------------------------------------------------------------===//

#include "Config/dpctl_config.h"
#include "dpctl_utils_helper.h"
#include <gtest/gtest.h>
#include <string>
#include <sycl/sycl.hpp>

struct TestHelperFns : public ::testing::Test
{
};

TEST_F(TestHelperFns, ChkDeviceTypeToStr)
{
    std::string res;
    EXPECT_NO_FATAL_FAILURE(
        res = DPCTL_DeviceTypeToStr(sycl::info::device_type::cpu));
    ASSERT_TRUE(res == "cpu");

    EXPECT_NO_FATAL_FAILURE(
        res = DPCTL_DeviceTypeToStr(sycl::info::device_type::gpu));
    ASSERT_TRUE(res == "gpu");

    EXPECT_NO_FATAL_FAILURE(
        res = DPCTL_DeviceTypeToStr(sycl::info::device_type::custom));
    ASSERT_TRUE(res == "custom");

    EXPECT_NO_FATAL_FAILURE(
        res = DPCTL_DeviceTypeToStr(sycl::info::device_type::accelerator));
    ASSERT_TRUE(res == "accelerator");

    EXPECT_NO_FATAL_FAILURE(
        res = DPCTL_DeviceTypeToStr(sycl::info::device_type::all));
    ASSERT_TRUE(res == "unknown");
}

TEST_F(TestHelperFns, ChkStrToDeviceType)
{
    sycl::info::device_type dev_type = sycl::info::device_type::automatic;

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_StrToDeviceType("cpu"));
    ASSERT_TRUE(dev_type == sycl::info::device_type::cpu);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_StrToDeviceType("gpu"));
    ASSERT_TRUE(dev_type == sycl::info::device_type::gpu);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_StrToDeviceType("accelerator"));
    ASSERT_TRUE(dev_type == sycl::info::device_type::accelerator);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_StrToDeviceType("custom"));
    ASSERT_TRUE(dev_type == sycl::info::device_type::custom);

    EXPECT_THROW(DPCTL_StrToDeviceType("invalid"), std::runtime_error);
}

TEST_F(TestHelperFns, ChkDPCTLBackendTypeToSyclBackend)
{
    sycl::backend res = sycl::backend::ext_oneapi_level_zero;

    EXPECT_NO_FATAL_FAILURE(res = DPCTL_DPCTLBackendTypeToSyclBackend(
                                DPCTLSyclBackendType::DPCTL_CUDA));
    ASSERT_TRUE(res == sycl::backend::ext_oneapi_cuda);

    EXPECT_NO_FATAL_FAILURE(res = DPCTL_DPCTLBackendTypeToSyclBackend(
                                DPCTLSyclBackendType::DPCTL_HIP));
    ASSERT_TRUE(res == sycl::backend::ext_oneapi_hip);

    EXPECT_NO_FATAL_FAILURE(res = DPCTL_DPCTLBackendTypeToSyclBackend(
                                DPCTLSyclBackendType::DPCTL_OPENCL));
    ASSERT_TRUE(res == sycl::backend::opencl);

    EXPECT_NO_FATAL_FAILURE(res = DPCTL_DPCTLBackendTypeToSyclBackend(
                                DPCTLSyclBackendType::DPCTL_LEVEL_ZERO));
    ASSERT_TRUE(res == sycl::backend::ext_oneapi_level_zero);

    EXPECT_THROW(DPCTL_DPCTLBackendTypeToSyclBackend(
                     DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND),
                 std::runtime_error);
}

TEST_F(TestHelperFns, ChkSyclBackendToDPCTLBackendType)
{
    DPCTLSyclBackendType DTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclBackendToDPCTLBackendType(
                                sycl::backend::ext_oneapi_level_zero));
    ASSERT_TRUE(DTy == DPCTLSyclBackendType::DPCTL_LEVEL_ZERO);

    EXPECT_NO_FATAL_FAILURE(
        DTy = DPCTL_SyclBackendToDPCTLBackendType(sycl::backend::opencl));
    ASSERT_TRUE(DTy == DPCTLSyclBackendType::DPCTL_OPENCL);

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclBackendToDPCTLBackendType(
                                sycl::backend::ext_oneapi_cuda));
    ASSERT_TRUE(DTy == DPCTLSyclBackendType::DPCTL_CUDA);

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclBackendToDPCTLBackendType(
                                sycl::backend::ext_oneapi_hip));
    ASSERT_TRUE(DTy == DPCTLSyclBackendType::DPCTL_HIP);

    EXPECT_NO_FATAL_FAILURE(
        DTy = DPCTL_SyclBackendToDPCTLBackendType(sycl::backend::all));
    ASSERT_TRUE(DTy == DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND);
}

TEST_F(TestHelperFns, ChkDPCTLDeviceTypeToSyclDeviceType)
{
    sycl::info::device_type dev_type = sycl::info::device_type::automatic;

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_DPCTLDeviceTypeToSyclDeviceType(
                                DPCTLSyclDeviceType::DPCTL_CPU));
    ASSERT_TRUE(dev_type == sycl::info::device_type::cpu);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_DPCTLDeviceTypeToSyclDeviceType(
                                DPCTLSyclDeviceType::DPCTL_GPU));
    ASSERT_TRUE(dev_type == sycl::info::device_type::gpu);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_DPCTLDeviceTypeToSyclDeviceType(
                                DPCTLSyclDeviceType::DPCTL_ACCELERATOR));
    ASSERT_TRUE(dev_type == sycl::info::device_type::accelerator);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_DPCTLDeviceTypeToSyclDeviceType(
                                DPCTLSyclDeviceType::DPCTL_CUSTOM));
    ASSERT_TRUE(dev_type == sycl::info::device_type::custom);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_DPCTLDeviceTypeToSyclDeviceType(
                                DPCTLSyclDeviceType::DPCTL_AUTOMATIC));
    ASSERT_TRUE(dev_type == sycl::info::device_type::automatic);

    EXPECT_NO_FATAL_FAILURE(dev_type = DPCTL_DPCTLDeviceTypeToSyclDeviceType(
                                DPCTLSyclDeviceType::DPCTL_ALL));
    ASSERT_TRUE(dev_type == sycl::info::device_type::all);
}

TEST_F(TestHelperFns, SyclDeviceTypeToDPCTLDeviceType)
{
    DPCTLSyclDeviceType DTy = DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE;

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(
                                sycl::info::device_type::cpu));
    ASSERT_TRUE(DTy == DPCTLSyclDeviceType::DPCTL_CPU);

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(
                                sycl::info::device_type::gpu));
    ASSERT_TRUE(DTy == DPCTLSyclDeviceType::DPCTL_GPU);

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(
                                sycl::info::device_type::accelerator));
    ASSERT_TRUE(DTy == DPCTLSyclDeviceType::DPCTL_ACCELERATOR);

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(
                                sycl::info::device_type::automatic));
    ASSERT_TRUE(DTy == DPCTLSyclDeviceType::DPCTL_AUTOMATIC);

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(
                                sycl::info::device_type::all));
    ASSERT_TRUE(DTy == DPCTLSyclDeviceType::DPCTL_ALL);

    EXPECT_NO_FATAL_FAILURE(DTy = DPCTL_SyclDeviceTypeToDPCTLDeviceType(
                                sycl::info::device_type::custom));
    ASSERT_TRUE(DTy == DPCTLSyclDeviceType::DPCTL_CUSTOM);
}

TEST_F(TestHelperFns, ChkDPCTLPeerAccessTypeToSycl)
{
    sycl::ext::oneapi::peer_access peer_type =
        sycl::ext::oneapi::peer_access::atomics_supported;

    EXPECT_NO_FATAL_FAILURE(peer_type = DPCTL_DPCTLPeerAccessTypeToSycl(
                                DPCTLPeerAccessType::access_supported));
    ASSERT_TRUE(peer_type == sycl::ext::oneapi::peer_access::access_supported);

    EXPECT_NO_FATAL_FAILURE(peer_type = DPCTL_DPCTLPeerAccessTypeToSycl(
                                DPCTLPeerAccessType::atomics_supported));
    ASSERT_TRUE(peer_type == sycl::ext::oneapi::peer_access::atomics_supported);
}

TEST_F(TestHelperFns, ChkSyclPeerAccessToDPCTLType)
{
    DPCTLPeerAccessType PTy = DPCTLPeerAccessType::atomics_supported;

    EXPECT_NO_FATAL_FAILURE(
        PTy = DPCTL_SyclPeerAccessToDPCTLType(
            sycl::ext::oneapi::peer_access::access_supported));
    ASSERT_TRUE(PTy == DPCTLPeerAccessType::access_supported);

    EXPECT_NO_FATAL_FAILURE(
        PTy = DPCTL_SyclPeerAccessToDPCTLType(
            sycl::ext::oneapi::peer_access::atomics_supported));
    ASSERT_TRUE(PTy == DPCTLPeerAccessType::atomics_supported);
}
