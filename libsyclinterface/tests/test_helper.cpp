//===--- test_helper.cpp - Test cases for helper functions  ===//
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
/// helper/include/dpctl_utils_helper.h.
///
//===----------------------------------------------------------------------===//

#include "Config/dpctl_config.h"
#include "dpctl_error_handlers.h"
#include "dpctl_utils_helper.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>

struct TestHelperFns : public ::testing::Test
{
};

namespace
{

struct AsyncErrors
{
    int count = 0;
    int last_code = 0;
    std::string last_msg;
    int released = 0;
};

void record_async_error(int err_code, const char *err_msg, void *user_data)
{
    auto errors = static_cast<AsyncErrors *>(user_data);
    ++errors->count;
    errors->last_code = err_code;
    errors->last_msg = err_msg;
}

void record_release(void *user_data)
{
    ++static_cast<AsyncErrors *>(user_data)->released;
}

template <typename ExceptionT>
void throw_from_host_task(sycl::queue &q, ExceptionT exc)
{
    q.submit(
        [&](sycl::handler &cgh) { cgh.host_task([exc]() { throw exc; }); });
}

} // namespace

TEST_F(TestHelperFns, ChkAsyncErrorHandlerPassesSyclException)
{
    AsyncErrors errors;
    DPCTLAsyncErrorHandlerSpec spec{&record_async_error, &errors, nullptr};
    sycl::queue q(sycl::default_selector_v, DPCTL_AsyncErrorHandler(&spec));

    throw_from_host_task(
        q, sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                           "async sycl error"));
    EXPECT_NO_THROW(q.wait_and_throw());

    EXPECT_EQ(errors.count, 1);
    EXPECT_EQ(errors.last_code, static_cast<int>(sycl::errc::invalid));
    EXPECT_NE(errors.last_msg.find("async sycl error"), std::string::npos);
}

TEST_F(TestHelperFns, ChkAsyncErrorHandlerPassesOtherException)
{
    AsyncErrors errors;
    DPCTLAsyncErrorHandlerSpec spec{&record_async_error, &errors, nullptr};
    sycl::queue q(sycl::default_selector_v, DPCTL_AsyncErrorHandler(&spec));

    throw_from_host_task(q, std::runtime_error("async std error"));
    EXPECT_NO_THROW(q.wait_and_throw());

    EXPECT_EQ(errors.count, 1);
    EXPECT_EQ(errors.last_code, -1);
    EXPECT_EQ(errors.last_msg, "async std error");
}

TEST_F(TestHelperFns, ChkAsyncErrorHandlerToleratesNullSpec)
{
    sycl::queue q(sycl::default_selector_v, DPCTL_AsyncErrorHandler(nullptr));

    throw_from_host_task(q, std::runtime_error("async std error"));
    EXPECT_NO_THROW(q.wait_and_throw());
}

TEST_F(TestHelperFns, ChkAsyncErrorHandlerToleratesNullCallback)
{
    AsyncErrors errors;
    DPCTLAsyncErrorHandlerSpec spec{nullptr, &errors, nullptr};
    sycl::queue q(sycl::default_selector_v, DPCTL_AsyncErrorHandler(&spec));

    throw_from_host_task(q, std::runtime_error("async std error"));
    EXPECT_NO_THROW(q.wait_and_throw());
    EXPECT_EQ(errors.count, 0);
}

TEST_F(TestHelperFns, ChkAsyncErrorHandlerReleasesOnceForAllCopies)
{
    AsyncErrors errors;
    DPCTLAsyncErrorHandlerSpec spec{&record_async_error, &errors,
                                    &record_release};
    // no work is submitted, as the runtime can hold a queue until it has
    // cleaned up after its work
    {
        DPCTL_AsyncErrorHandler handler(&spec);
        auto handler_copy = handler;
        sycl::queue q(sycl::default_selector_v, handler);
        auto q_copy = q;
        sycl::queue q2(q.get_context(), q.get_device(), handler_copy);
        q.wait();
        q2.wait();
        EXPECT_EQ(errors.released, 0);
    }
    EXPECT_EQ(errors.released, 1);
}

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
