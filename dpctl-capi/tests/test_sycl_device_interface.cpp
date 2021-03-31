//===--- test_sycl_device_interface.cpp - Test cases for device interface  ===//
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
/// dpctl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//

#include "../helper/include/dpctl_utils_helper.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_utils.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

struct TestDPCTLSyclDeviceInterface
    : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceRef DRef = nullptr;

    TestDPCTLSyclDeviceInterface()
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

    ~TestDPCTLSyclDeviceInterface()
    {
        DPCTLDevice_Delete(DRef);
    }
};

TEST_P(TestDPCTLSyclDeviceInterface, Chk_Copy)
{
    DPCTLSyclDeviceRef Copied_DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_DRef = DPCTLDevice_Copy(DRef));
    EXPECT_TRUE(bool(Copied_DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(Copied_DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetBackend)
{
    DPCTLSyclBackendType BTy = DPCTLSyclBackendType::DPCTL_UNKNOWN_BACKEND;
    EXPECT_NO_FATAL_FAILURE(BTy = DPCTLDevice_GetBackend(DRef));
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

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetDeviceType)
{
    DPCTLSyclDeviceType DTy = DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE;
    EXPECT_NO_FATAL_FAILURE(DTy = DPCTLDevice_GetDeviceType(DRef));
    EXPECT_TRUE(DTy != DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE);
    EXPECT_TRUE(DTy != DPCTLSyclDeviceType::DPCTL_ALL);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetDriverInfo)
{
    const char *DriverInfo = nullptr;
    EXPECT_NO_FATAL_FAILURE(DriverInfo = DPCTLDevice_GetDriverInfo(DRef));
    EXPECT_TRUE(DriverInfo != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(DriverInfo));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetName)
{
    const char *Name = nullptr;
    EXPECT_NO_FATAL_FAILURE(Name = DPCTLDevice_GetName(DRef));
    EXPECT_TRUE(Name != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(Name));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetVendorName)
{
    const char *VendorName = nullptr;
    EXPECT_NO_FATAL_FAILURE(VendorName = DPCTLDevice_GetVendorName(DRef));
    EXPECT_TRUE(VendorName != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(VendorName));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetMaxComputeUnits)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxComputeUnits(DRef));
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetMaxWorkItemDims)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxWorkItemDims(DRef));
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetMaxWorkItemSizes)
{
    size_t *sizes = nullptr;
    EXPECT_NO_FATAL_FAILURE(sizes = DPCTLDevice_GetMaxWorkItemSizes(DRef));
    EXPECT_TRUE(sizes != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLSize_t_Array_Delete(sizes));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetMaxWorkGroupSize)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxWorkGroupSize(DRef));
    if (DPCTLDevice_IsAccelerator(DRef))
        EXPECT_TRUE(n >= 0);
    else
        EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetMaxNumSubGroups)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxNumSubGroups(DRef));
    if (DPCTLDevice_IsAccelerator(DRef))
        EXPECT_TRUE(n >= 0);
    else
        EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPlatform)
{
    DPCTLSyclPlatformRef PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(PRef = DPCTLDevice_GetPlatform(DRef));
    ASSERT_TRUE(PRef);
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(PRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_IsAccelerator)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsAccelerator(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_IsCPU)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsCPU(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_IsGPU)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsGPU(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_IsHost)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsHost(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetSubGroupIndependentForwardProgress)
{
    bool sub_group_progress = 0;
    EXPECT_NO_FATAL_FAILURE(
        sub_group_progress =
            DPCTLDevice_GetSubGroupIndependentForwardProgress(DRef));
    auto D = reinterpret_cast<device *>(DRef);
    auto get_sub_group_progress =
        D->get_info<info::device::sub_group_independent_forward_progress>();
    EXPECT_TRUE(get_sub_group_progress == sub_group_progress);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPreferredVectorWidthChar)
{
    size_t vector_width_char = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_char =
                                DPCTLDevice_GetPreferredVectorWidthChar(DRef));
    EXPECT_TRUE(vector_width_char != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPreferredVectorWidthShort)
{
    size_t vector_width_short = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_short =
                                DPCTLDevice_GetPreferredVectorWidthShort(DRef));
    EXPECT_TRUE(vector_width_short != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPreferredVectorWidthInt)
{
    size_t vector_width_int = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_int =
                                DPCTLDevice_GetPreferredVectorWidthInt(DRef));
    EXPECT_TRUE(vector_width_int != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPreferredVectorWidthLong)
{
    size_t vector_width_long = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_long =
                                DPCTLDevice_GetPreferredVectorWidthLong(DRef));
    EXPECT_TRUE(vector_width_long != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPreferredVectorWidthFloat)
{
    size_t vector_width_float = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_float =
                                DPCTLDevice_GetPreferredVectorWidthFloat(DRef));
    EXPECT_TRUE(vector_width_float != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPreferredVectorWidthDouble)
{
    size_t vector_width_double = 0;
    EXPECT_NO_FATAL_FAILURE(
        vector_width_double = DPCTLDevice_GetPreferredVectorWidthDouble(DRef));
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("fp64"))))
    {
        EXPECT_TRUE(vector_width_double != 0);
    }
    else {
        EXPECT_TRUE(vector_width_double == 0);
    }
}

TEST_P(TestDPCTLSyclDeviceInterface, Chk_GetPreferredVectorWidthHalf)
{
    size_t vector_width_half = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_half =
                                DPCTLDevice_GetPreferredVectorWidthHalf(DRef));
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("fp16"))))
    {
        EXPECT_TRUE(vector_width_half != 0);
    }
    else {
        EXPECT_TRUE(vector_width_half == 0);
    }
}

INSTANTIATE_TEST_SUITE_P(DPCTLDevice_Fns,
                         TestDPCTLSyclDeviceInterface,
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
