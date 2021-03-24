//===--- test_sycl_context_interface.cpp - Test cases for device interface ===//
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
/// dpctl_sycl_context_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_types.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace
{
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(sycl::device, DPCTLSyclDeviceRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(vector_class<DPCTLSyclDeviceRef>,
                                   DPCTLDeviceVectorRef)
} // namespace

struct TestDPCTLContextInterface : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;

    TestDPCTLContextInterface()
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

    ~TestDPCTLContextInterface()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
};

TEST_P(TestDPCTLContextInterface, Chk_Create)
{
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef);
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_P(TestDPCTLContextInterface, Chk_CreateWithDevices)
{
    size_t nCUs = 0;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    /* TODO: Once we have wrappers for sub-device creation let us use those
     * functions.
     */
    EXPECT_NO_FATAL_FAILURE(nCUs = DPCTLDevice_GetMaxComputeUnits(DRef));
    if (nCUs) {
        auto D = unwrap(DRef);
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_equally>(nCUs / 2);
            EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLDeviceVector_Create());
            for (auto &sd : subDevices) {
                unwrap(DVRef)->emplace_back(wrap(new device(sd)));
            }
            EXPECT_NO_FATAL_FAILURE(
                CRef = DPCTLContext_CreateFromDevices(DVRef, nullptr, 0));
            ASSERT_TRUE(CRef);
        } catch (feature_not_supported const &fnse) {
            GTEST_SKIP_("Skipping creating context for sub-devices");
        }
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_P(TestDPCTLContextInterface, Chk_CreateWithDevices2)
{
    size_t nCUs = 0;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLDeviceVectorRef DVRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");

    /* TODO: Once we have wrappers for sub-device creation let us use those
     * functions.
     */
    EXPECT_NO_FATAL_FAILURE(nCUs = DPCTLDevice_GetMaxComputeUnits(DRef));
    if (nCUs) {
        auto D = unwrap(DRef);
        try {
            auto subDevices = D->create_sub_devices<
                info::partition_property::partition_equally>(nCUs / 2);
	    const size_t len = subDevices.size();
	    auto ar = new DPCTLSyclDeviceRef[len];
	    for(size_t i=0; i < len; ++i) {
		ar[i] = wrap(new device(subDevices.at(i)));
	    }
            EXPECT_NO_FATAL_FAILURE(DVRef = DPCTLDeviceVector_CreateFromArray(len, ar));
            EXPECT_NO_FATAL_FAILURE(
                CRef = DPCTLContext_CreateFromDevices(DVRef, nullptr, 0));
            ASSERT_TRUE(CRef);
	    delete[] ar;
        } catch (feature_not_supported const &fnse) {
            GTEST_SKIP_("Skipping creating context for sub-devices");
        }
    }
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDeviceVector_Delete(DVRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_P(TestDPCTLContextInterface, Chk_AreEq)
{
    DPCTLSyclContextRef CRef1 = nullptr, CRef2 = nullptr, CRef3 = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    bool are_eq = true, are_not_eq = false;

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");
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

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef1));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef2));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef3));
}

TEST_P(TestDPCTLContextInterface, Chk_IsHost)
{
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    bool is_host_device = false, is_host_context = false;

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef);

    EXPECT_NO_FATAL_FAILURE(is_host_device = DPCTLDevice_IsHost(DRef));
    EXPECT_NO_FATAL_FAILURE(is_host_context = DPCTLContext_IsHost(CRef));
    EXPECT_TRUE(is_host_device == is_host_context);

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLContext_Delete(CRef));
}

TEST_P(TestDPCTLContextInterface, Chk_GetBackend)
{
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclBackendType context_backend = DPCTL_UNKNOWN_BACKEND,
                         device_backend = DPCTL_UNKNOWN_BACKEND;

    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    if (!DRef)
        GTEST_SKIP_("Device not found");
    EXPECT_NO_FATAL_FAILURE(CRef = DPCTLContext_Create(DRef, nullptr, 0));
    ASSERT_TRUE(CRef);

    EXPECT_NO_FATAL_FAILURE(device_backend = DPCTLDevice_GetBackend(DRef));
    EXPECT_NO_FATAL_FAILURE(context_backend = DPCTLContext_GetBackend(CRef));
    EXPECT_TRUE(device_backend == context_backend);

    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
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
