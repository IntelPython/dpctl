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

TEST_P(TestDPCTLSyclDeviceInterface, ChkCopy)
{
    DPCTLSyclDeviceRef Copied_DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_DRef = DPCTLDevice_Copy(DRef));
    EXPECT_TRUE(bool(Copied_DRef));
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(Copied_DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetBackend)
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

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetDeviceType)
{
    DPCTLSyclDeviceType DTy = DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE;
    EXPECT_NO_FATAL_FAILURE(DTy = DPCTLDevice_GetDeviceType(DRef));
    EXPECT_TRUE(DTy != DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE);
    EXPECT_TRUE(DTy != DPCTLSyclDeviceType::DPCTL_ALL);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetDriverInfo)
{
    const char *DriverInfo = nullptr;
    EXPECT_NO_FATAL_FAILURE(DriverInfo = DPCTLDevice_GetDriverInfo(DRef));
    EXPECT_TRUE(DriverInfo != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(DriverInfo));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetName)
{
    const char *Name = nullptr;
    EXPECT_NO_FATAL_FAILURE(Name = DPCTLDevice_GetName(DRef));
    EXPECT_TRUE(Name != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(Name));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetVendorName)
{
    const char *VendorName = nullptr;
    EXPECT_NO_FATAL_FAILURE(VendorName = DPCTLDevice_GetVendorName(DRef));
    EXPECT_TRUE(VendorName != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(VendorName));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxComputeUnits)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxComputeUnits(DRef));
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxWorkItemDims)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxWorkItemDims(DRef));
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxWorkItemSizes)
{
    size_t *sizes = nullptr;
    EXPECT_NO_FATAL_FAILURE(sizes = DPCTLDevice_GetMaxWorkItemSizes(DRef));
    EXPECT_TRUE(sizes != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLSize_t_Array_Delete(sizes));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxWorkGroupSize)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxWorkGroupSize(DRef));
    if (DPCTLDevice_IsAccelerator(DRef))
        EXPECT_TRUE(n >= 0);
    else
        EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxNumSubGroups)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxNumSubGroups(DRef));
    if (DPCTLDevice_IsAccelerator(DRef))
        EXPECT_TRUE(n >= 0);
    else
        EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPlatform)
{
    DPCTLSyclPlatformRef PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(PRef = DPCTLDevice_GetPlatform(DRef));
    ASSERT_TRUE(PRef);
    EXPECT_NO_FATAL_FAILURE(DPCTLPlatform_Delete(PRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkIsAccelerator)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsAccelerator(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkIsCPU)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsCPU(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkIsGPU)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsGPU(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkIsHost)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_IsHost(DRef));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetSubGroupIndependentForwardProgress)
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

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPreferredVectorWidthChar)
{
    size_t vector_width_char = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_char =
                                DPCTLDevice_GetPreferredVectorWidthChar(DRef));
    EXPECT_TRUE(vector_width_char != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPreferredVectorWidthShort)
{
    size_t vector_width_short = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_short =
                                DPCTLDevice_GetPreferredVectorWidthShort(DRef));
    EXPECT_TRUE(vector_width_short != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPreferredVectorWidthInt)
{
    size_t vector_width_int = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_int =
                                DPCTLDevice_GetPreferredVectorWidthInt(DRef));
    EXPECT_TRUE(vector_width_int != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPreferredVectorWidthLong)
{
    size_t vector_width_long = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_long =
                                DPCTLDevice_GetPreferredVectorWidthLong(DRef));
    EXPECT_TRUE(vector_width_long != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPreferredVectorWidthFloat)
{
    size_t vector_width_float = 0;
    EXPECT_NO_FATAL_FAILURE(vector_width_float =
                                DPCTLDevice_GetPreferredVectorWidthFloat(DRef));
    EXPECT_TRUE(vector_width_float != 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPreferredVectorWidthDouble)
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

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetPreferredVectorWidthHalf)
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

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxReadImageArgs)
{
    size_t max_read_image_args = 0;
    EXPECT_NO_FATAL_FAILURE(max_read_image_args =
                                DPCTLDevice_GetMaxReadImageArgs(DRef));
    size_t min_val = 128;
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("image"))))
        EXPECT_TRUE(max_read_image_args >= min_val);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxWriteImageArgs)
{
    size_t max_write_image_args = 0;
    EXPECT_NO_FATAL_FAILURE(max_write_image_args =
                                DPCTLDevice_GetMaxWriteImageArgs(DRef));
    size_t min_val = 8;
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("image"))))
        EXPECT_TRUE(max_write_image_args >= min_val);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetImage2dMaxWidth)
{
    size_t image_2d_max_width = 0;
    EXPECT_NO_FATAL_FAILURE(image_2d_max_width =
                                DPCTLDevice_GetImage2dMaxWidth(DRef));
    size_t min_val = 8192;
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("image"))))
        EXPECT_TRUE(image_2d_max_width >= min_val);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetImage2dMaxHeight)
{
    size_t image_2d_max_height = 0;
    EXPECT_NO_FATAL_FAILURE(image_2d_max_height =
                                DPCTLDevice_GetImage2dMaxHeight(DRef));
    size_t min_val = 8192;
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("image"))))
        EXPECT_TRUE(image_2d_max_height >= min_val);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetImage3dMaxWidth)
{
    size_t image_3d_max_width = 0;
    EXPECT_NO_FATAL_FAILURE(image_3d_max_width =
                                DPCTLDevice_GetImage3dMaxWidth(DRef));
    size_t min_val = 2048;
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("image"))))
        EXPECT_TRUE(image_3d_max_width >= min_val);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetImage3dMaxHeight)
{
    size_t image_3d_max_height = 0;
    EXPECT_NO_FATAL_FAILURE(image_3d_max_height =
                                DPCTLDevice_GetImage3dMaxHeight(DRef));
    size_t min_val = 2048;
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("image"))))
        EXPECT_TRUE(image_3d_max_height >= min_val);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetImage3dMaxDepth)
{
    size_t image_3d_max_depth = 0;
    EXPECT_NO_FATAL_FAILURE(image_3d_max_depth =
                                DPCTLDevice_GetImage3dMaxDepth(DRef));
    size_t min_val = 2048;
    if (DPCTLDevice_HasAspect(DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                        DPCTL_StrToAspectType("image"))))
        EXPECT_TRUE(image_3d_max_depth >= min_val);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetParentDevice)
{
    DPCTLSyclDeviceRef pDRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(pDRef = DPCTLDevice_GetParentDevice(DRef));
    EXPECT_TRUE(pDRef == nullptr);
}

INSTANTIATE_TEST_SUITE_P(DPCTLDeviceFns,
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
