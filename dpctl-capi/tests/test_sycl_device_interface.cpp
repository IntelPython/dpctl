//===--- test_sycl_device_interface.cpp - Test cases for device interface  ===//
//
//                      Data Parallel Control (dpCtl)
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
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "dpctl_utils.h"
#include <CL/sycl.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

struct TestDPCTLSyclDeviceInterface : public ::testing::Test
{
    DPCTLSyclDeviceRef OpenCL_cpu = nullptr;
    DPCTLSyclDeviceRef OpenCL_gpu = nullptr;
    DPCTLSyclDeviceRef OpenCL_Level0_gpu = nullptr;

    TestDPCTLSyclDeviceInterface()
    {
        if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_CPU)) {
            auto Q = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_CPU, 0);
            OpenCL_cpu = DPCTLQueue_GetDevice(Q);
            DPCTLQueue_Delete(Q);
        }

        if (DPCTLQueueMgr_GetNumQueues(DPCTL_OPENCL, DPCTL_GPU)) {
            auto Q = DPCTLQueueMgr_GetQueue(DPCTL_OPENCL, DPCTL_GPU, 0);
            OpenCL_gpu = DPCTLQueue_GetDevice(Q);
            DPCTLQueue_Delete(Q);
        }

        if (DPCTLQueueMgr_GetNumQueues(DPCTL_LEVEL_ZERO, DPCTL_GPU)) {
            auto Q = DPCTLQueueMgr_GetQueue(DPCTL_LEVEL_ZERO, DPCTL_GPU, 0);
            OpenCL_Level0_gpu = DPCTLQueue_GetDevice(Q);
            DPCTLQueue_Delete(Q);
        }
    }

    ~TestDPCTLSyclDeviceInterface()
    {
        DPCTLDevice_Delete(OpenCL_cpu);
        DPCTLDevice_Delete(OpenCL_gpu);
        DPCTLDevice_Delete(OpenCL_Level0_gpu);
    }
};

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetDriverInfo)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DriverInfo = DPCTLDevice_GetDriverInfo(OpenCL_cpu);
    EXPECT_TRUE(DriverInfo != nullptr);
    DPCTLCString_Delete(DriverInfo);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetDriverInfo)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DriverInfo = DPCTLDevice_GetDriverInfo(OpenCL_gpu);
    EXPECT_TRUE(DriverInfo != nullptr);
    DPCTLCString_Delete(DriverInfo);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetDriverInfo)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto DriverInfo = DPCTLDevice_GetDriverInfo(OpenCL_Level0_gpu);
    EXPECT_TRUE(DriverInfo != nullptr);
    DPCTLCString_Delete(DriverInfo);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetMaxComputeUnits)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPCTLDevice_GetMaxComputeUnits(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetMaxComputeUnits)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPCTLDevice_GetMaxComputeUnits(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetMaxComputeUnits)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPCTLDevice_GetMaxComputeUnits(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetMaxWorkItemDims)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPCTLDevice_GetMaxWorkItemDims(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetMaxWorkItemDims)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPCTLDevice_GetMaxWorkItemDims(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetMaxWorkItemDims)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPCTLDevice_GetMaxWorkItemDims(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetMaxWorkItemSizes)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto item_sizes = DPCTLDevice_GetMaxWorkItemSizes(OpenCL_cpu);
    EXPECT_TRUE(item_sizes != nullptr);
    DPCTLSize_t_Array_Delete(item_sizes);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetMaxWorkItemSizes)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto item_sizes = DPCTLDevice_GetMaxWorkItemSizes(OpenCL_gpu);
    EXPECT_TRUE(item_sizes != nullptr);
    DPCTLSize_t_Array_Delete(item_sizes);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetMaxWorkItemSizes)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto item_sizes = DPCTLDevice_GetMaxWorkItemSizes(OpenCL_Level0_gpu);
    EXPECT_TRUE(item_sizes != nullptr);
    DPCTLSize_t_Array_Delete(item_sizes);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetMaxWorkGroupSize)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPCTLDevice_GetMaxWorkGroupSize(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetMaxWorkGroupSize)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPCTLDevice_GetMaxWorkGroupSize(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetMaxWorkGroupSize)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPCTLDevice_GetMaxWorkGroupSize(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetMaxNumSubGroups)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPCTLDevice_GetMaxNumSubGroups(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetMaxNumSubGroups)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPCTLDevice_GetMaxNumSubGroups(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetMaxNumSubGroups)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPCTLDevice_GetMaxNumSubGroups(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

// TODO: Update when DPC++ properly supports aspects
TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_HasInt64BaseAtomics)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto atomics = DPCTLDevice_HasInt64BaseAtomics(OpenCL_cpu);
    auto D = reinterpret_cast<device *>(OpenCL_cpu);
    auto has_atomics = D->has(aspect::int64_base_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

// TODO: Update when DPC++ properly supports aspects
TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_HasInt64BaseAtomics)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto atomics = DPCTLDevice_HasInt64BaseAtomics(OpenCL_gpu);
    auto D = reinterpret_cast<device *>(OpenCL_gpu);
    auto has_atomics = D->has(aspect::int64_base_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

// TODO: Update when DPC++ properly supports aspects
TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_HasInt64BaseAtomics)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto atomics = DPCTLDevice_HasInt64BaseAtomics(OpenCL_Level0_gpu);
    auto D = reinterpret_cast<device *>(OpenCL_Level0_gpu);
    auto has_atomics = D->has(aspect::int64_base_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

// TODO: Update when DPC++ properly supports aspects
TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_HasInt64ExtendedAtomics)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto atomics = DPCTLDevice_HasInt64ExtendedAtomics(OpenCL_cpu);
    auto D = reinterpret_cast<device *>(OpenCL_cpu);
    auto has_atomics = D->has(aspect::int64_extended_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

// TODO: Update when DPC++ properly supports aspects
TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_HasInt64ExtendedAtomics)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto atomics = DPCTLDevice_HasInt64ExtendedAtomics(OpenCL_gpu);
    auto D = reinterpret_cast<device *>(OpenCL_gpu);
    auto has_atomics = D->has(aspect::int64_extended_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

// TODO: Update when DPC++ properly supports aspects
TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_HasInt64ExtendedAtomics)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto atomics = DPCTLDevice_HasInt64ExtendedAtomics(OpenCL_Level0_gpu);
    auto D = reinterpret_cast<device *>(OpenCL_Level0_gpu);
    auto has_atomics = D->has(aspect::int64_extended_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetName)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DevName = DPCTLDevice_GetName(OpenCL_cpu);
    EXPECT_TRUE(DevName != nullptr);
    DPCTLCString_Delete(DevName);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetName)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DevName = DPCTLDevice_GetName(OpenCL_gpu);
    EXPECT_TRUE(DevName != nullptr);
    DPCTLCString_Delete(DevName);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetName)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto DevName = DPCTLDevice_GetName(OpenCL_Level0_gpu);
    EXPECT_TRUE(DevName != nullptr);
    DPCTLCString_Delete(DevName);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_GetVendorName)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto VendorName = DPCTLDevice_GetVendorName(OpenCL_cpu);
    EXPECT_TRUE(VendorName != nullptr);
    DPCTLCString_Delete(VendorName);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_GetVendorName)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto VendorName = DPCTLDevice_GetVendorName(OpenCL_gpu);
    EXPECT_TRUE(VendorName != nullptr);
    DPCTLCString_Delete(VendorName);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_GetVendorName)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto VendorName = DPCTLDevice_GetVendorName(OpenCL_Level0_gpu);
    EXPECT_TRUE(VendorName != nullptr);
    DPCTLCString_Delete(VendorName);
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLCPU_IsCPU)
{
    if (!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    EXPECT_TRUE(DPCTLDevice_IsCPU(OpenCL_cpu));
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckOCLGPU_IsGPU)
{
    if (!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    EXPECT_TRUE(DPCTLDevice_IsGPU(OpenCL_gpu));
}

TEST_F(TestDPCTLSyclDeviceInterface, CheckLevel0GPU_IsGPU)
{
    if (!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    EXPECT_TRUE(DPCTLDevice_IsGPU(OpenCL_Level0_gpu));
}
