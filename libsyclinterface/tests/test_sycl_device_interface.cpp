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

#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_platform_interface.h"
#include "dpctl_utils.h"
#include "dpctl_utils_helper.h"
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
    EXPECT_TRUE(DPCTLDevice_AreEq(DRef, Copied_DRef));
    EXPECT_TRUE(DPCTLDevice_Hash(DRef) == DPCTLDevice_Hash(Copied_DRef));
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
    EXPECT_NO_FATAL_FAILURE(DriverInfo = DPCTLDevice_GetDriverVersion(DRef));
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
    EXPECT_NO_FATAL_FAILURE(VendorName = DPCTLDevice_GetVendor(DRef));
    EXPECT_TRUE(VendorName != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLCString_Delete(VendorName));
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetMaxComputeUnits)
{
    size_t n = 0;
    EXPECT_NO_FATAL_FAILURE(n = DPCTLDevice_GetMaxComputeUnits(DRef));
    EXPECT_TRUE(n > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetGlobalMemSize)
{
    size_t gm_sz = 0;
    EXPECT_NO_FATAL_FAILURE(gm_sz = DPCTLDevice_GetGlobalMemSize(DRef));
    EXPECT_TRUE(gm_sz > 0);
}

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetLocalMemSize)
{
    size_t lm_sz = 0;
    EXPECT_NO_FATAL_FAILURE(lm_sz = DPCTLDevice_GetLocalMemSize(DRef));
    EXPECT_TRUE(lm_sz > 0);
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

TEST_P(TestDPCTLSyclDeviceInterface, ChkGetProfilingTimerResolution)
{
    size_t res = 0;
    EXPECT_NO_FATAL_FAILURE(res =
                                DPCTLDevice_GetProfilingTimerResolution(DRef));
    EXPECT_TRUE(res != 0);
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

struct TestDPCTLSyclDeviceNullArgs : public ::testing::Test
{
    DPCTLSyclDeviceRef Null_DRef = nullptr;
    DPCTLSyclDeviceSelectorRef Null_DSRef = nullptr;
};

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkCopy)
{
    DPCTLSyclDeviceRef Copied_DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_DRef = DPCTLDevice_Copy(Null_DRef));
    ASSERT_FALSE(bool(Copied_DRef));
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkCreateFromSelector)
{
    DPCTLSyclDeviceRef Created_DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Created_DRef =
                                DPCTLDevice_CreateFromSelector(Null_DSRef));
    ASSERT_FALSE(bool(Created_DRef));
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkDelete)
{
    EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(Null_DRef));
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetDeviceType)
{
    DPCTLSyclDeviceType DTy = DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE;
    EXPECT_NO_FATAL_FAILURE(DTy = DPCTLDevice_GetDeviceType(Null_DRef));
    ASSERT_TRUE(DTy == DPCTLSyclDeviceType::DPCTL_UNKNOWN_DEVICE);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkIsAccelerator)
{
    bool is_acc = true;
    EXPECT_NO_FATAL_FAILURE(is_acc = DPCTLDevice_IsAccelerator(Null_DRef));
    ASSERT_FALSE(is_acc);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkIsCPU)
{
    bool is_cpu = true;
    EXPECT_NO_FATAL_FAILURE(is_cpu = DPCTLDevice_IsCPU(Null_DRef));
    ASSERT_FALSE(is_cpu);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkIsGPU)
{
    bool is_gpu = true;
    EXPECT_NO_FATAL_FAILURE(is_gpu = DPCTLDevice_IsGPU(Null_DRef));
    ASSERT_FALSE(is_gpu);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkIsHost)
{
    bool is_host = true;
    EXPECT_NO_FATAL_FAILURE(is_host = DPCTLDevice_IsHost(Null_DRef));
    ASSERT_FALSE(is_host);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetMaxComputeUnits)
{
    uint32_t mcu = -1;
    EXPECT_NO_FATAL_FAILURE(mcu = DPCTLDevice_GetMaxComputeUnits(Null_DRef));
    ASSERT_TRUE(mcu == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetGlobalMemSize)
{
    uint64_t gmsz = -1;
    EXPECT_NO_FATAL_FAILURE(gmsz = DPCTLDevice_GetGlobalMemSize(Null_DRef));
    ASSERT_TRUE(gmsz == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetLocalMemSize)
{
    uint64_t lmsz = -1;
    EXPECT_NO_FATAL_FAILURE(lmsz = DPCTLDevice_GetLocalMemSize(Null_DRef));
    ASSERT_TRUE(lmsz == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetMaxWorkItemDims)
{
    uint32_t md = -1;
    EXPECT_NO_FATAL_FAILURE(md = DPCTLDevice_GetMaxWorkItemDims(Null_DRef));
    ASSERT_TRUE(md == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetMaxWorkItemSizes)
{
    size_t *sz = nullptr;
    EXPECT_NO_FATAL_FAILURE(sz = DPCTLDevice_GetMaxWorkItemSizes(Null_DRef));
    ASSERT_TRUE(sz == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetMaxWorkGroupSize)
{
    size_t m = -1;
    EXPECT_NO_FATAL_FAILURE(m = DPCTLDevice_GetMaxWorkGroupSize(Null_DRef));
    ASSERT_TRUE(m == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetMaxNumSubGroups)
{
    uint32_t nsg = -1;
    EXPECT_NO_FATAL_FAILURE(nsg = DPCTLDevice_GetMaxNumSubGroups(Null_DRef));
    ASSERT_TRUE(nsg == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetPlatform)
{
    DPCTLSyclPlatformRef PRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(PRef = DPCTLDevice_GetPlatform(Null_DRef));
    ASSERT_TRUE(PRef == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkName)
{
    const char *name = nullptr;
    EXPECT_NO_FATAL_FAILURE(name = DPCTLDevice_GetName(Null_DRef));
    ASSERT_TRUE(name == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkVendor)
{
    const char *vendor = nullptr;
    EXPECT_NO_FATAL_FAILURE(vendor = DPCTLDevice_GetVendor(Null_DRef));
    ASSERT_TRUE(vendor == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkDriverVersion)
{
    const char *driver_version = nullptr;
    EXPECT_NO_FATAL_FAILURE(driver_version =
                                DPCTLDevice_GetDriverVersion(Null_DRef));
    ASSERT_TRUE(driver_version == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkIsHostUnifiedMemory)
{
    bool is_hum = true;
    EXPECT_NO_FATAL_FAILURE(is_hum =
                                DPCTLDevice_IsHostUnifiedMemory(Null_DRef));
    ASSERT_FALSE(is_hum);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkAreEq)
{
    bool are_eq = true;
    EXPECT_NO_FATAL_FAILURE(are_eq = DPCTLDevice_AreEq(Null_DRef, Null_DRef));
    ASSERT_FALSE(are_eq);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkHasAspect)
{
    bool has_fp64 = true;
    EXPECT_NO_FATAL_FAILURE(has_fp64 = DPCTLDevice_HasAspect(
                                Null_DRef, DPCTL_SyclAspectToDPCTLAspectType(
                                               DPCTL_StrToAspectType("fp64"))));
    ASSERT_FALSE(has_fp64);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetMaxReadWriteImageArgs)
{
    uint32_t res = 0;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLDevice_GetMaxReadImageArgs(Null_DRef));
    ASSERT_TRUE(res == 0);
    uint32_t wes = 0;
    EXPECT_NO_FATAL_FAILURE(wes = DPCTLDevice_GetMaxWriteImageArgs(Null_DRef));
    ASSERT_TRUE(wes == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetMaxImageDims)
{
    size_t res = 0;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLDevice_GetImage2dMaxWidth(Null_DRef));
    ASSERT_TRUE(res == 0);

    res = 0;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLDevice_GetImage2dMaxHeight(Null_DRef));
    ASSERT_TRUE(res == 0);

    res = 0;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLDevice_GetImage3dMaxHeight(Null_DRef));
    ASSERT_TRUE(res == 0);

    res = 0;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLDevice_GetImage3dMaxWidth(Null_DRef));
    ASSERT_TRUE(res == 0);

    res = 0;
    EXPECT_NO_FATAL_FAILURE(res = DPCTLDevice_GetImage3dMaxDepth(Null_DRef));
    ASSERT_TRUE(res == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetSubGroupIndependentForwardProgress)
{
    bool indep_pr = true;
    EXPECT_NO_FATAL_FAILURE(
        indep_pr =
            DPCTLDevice_GetSubGroupIndependentForwardProgress(Null_DRef));
    ASSERT_FALSE(indep_pr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetPreferredVectorWidth)
{
    uint32_t w = -1;
    EXPECT_NO_FATAL_FAILURE(
        w = DPCTLDevice_GetPreferredVectorWidthChar(Null_DRef));
    ASSERT_TRUE(w == 0);

    w = -1;
    EXPECT_NO_FATAL_FAILURE(
        w = DPCTLDevice_GetPreferredVectorWidthShort(Null_DRef));
    ASSERT_TRUE(w == 0);

    w = -1;
    EXPECT_NO_FATAL_FAILURE(
        w = DPCTLDevice_GetPreferredVectorWidthInt(Null_DRef));
    ASSERT_TRUE(w == 0);

    w = -1;
    EXPECT_NO_FATAL_FAILURE(
        w = DPCTLDevice_GetPreferredVectorWidthLong(Null_DRef));
    ASSERT_TRUE(w == 0);

    w = -1;
    EXPECT_NO_FATAL_FAILURE(
        w = DPCTLDevice_GetPreferredVectorWidthHalf(Null_DRef));
    ASSERT_TRUE(w == 0);

    w = -1;
    EXPECT_NO_FATAL_FAILURE(
        w = DPCTLDevice_GetPreferredVectorWidthFloat(Null_DRef));
    ASSERT_TRUE(w == 0);

    w = -1;
    EXPECT_NO_FATAL_FAILURE(
        w = DPCTLDevice_GetPreferredVectorWidthDouble(Null_DRef));
    ASSERT_TRUE(w == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetParentDevice)
{
    DPCTLSyclDeviceRef pDRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(pDRef = DPCTLDevice_GetParentDevice(Null_DRef));
    ASSERT_TRUE(pDRef == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkCreateSubDevicesEqually)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(
        DVRef = DPCTLDevice_CreateSubDevicesEqually(Null_DRef, 2));
    ASSERT_TRUE(DVRef == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkCreateSubDevicesByCounts)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    size_t counts[2] = {1, 1};
    EXPECT_NO_FATAL_FAILURE(
        DVRef = DPCTLDevice_CreateSubDevicesByCounts(Null_DRef, counts, 2));
    ASSERT_TRUE(DVRef == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkCreateSubDevicesByAffinity)
{
    DPCTLDeviceVectorRef DVRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(
        DVRef = DPCTLDevice_CreateSubDevicesByAffinity(
            Null_DRef, DPCTLPartitionAffinityDomainType::not_applicable));
    ASSERT_TRUE(DVRef == nullptr);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkHash)
{
    size_t hash = 0;
    EXPECT_NO_FATAL_FAILURE(hash = DPCTLDevice_Hash(Null_DRef));
    ASSERT_TRUE(hash == 0);
}

TEST_F(TestDPCTLSyclDeviceNullArgs, ChkGetProfilingTimerResolution)
{
    size_t res = 1;
    EXPECT_NO_FATAL_FAILURE(
        res = DPCTLDevice_GetProfilingTimerResolution(Null_DRef));
    ASSERT_TRUE(res == 0);
}
