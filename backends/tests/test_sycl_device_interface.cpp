//===----- test_sycl_device_interface.cpp - dpctl-C_API interface -*- C++ -*-===//
//
//               Python Data Parallel Processing Library (PyDPPL)
//
// Copyright 2020 Intel Corporation
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
/// dppl_sycl_device_interface.h.
///
//===----------------------------------------------------------------------===//

#include "dppl_sycl_device_interface.h"
#include "dppl_sycl_queue_interface.h"
#include "dppl_sycl_queue_manager.h"
#include "dppl_utils.h"

#include <gtest/gtest.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;


struct TestDPPLSyclDeviceInterface : public ::testing::Test
{
    DPPLSyclDeviceRef OpenCL_cpu = nullptr;
    DPPLSyclDeviceRef OpenCL_gpu = nullptr;
    DPPLSyclDeviceRef OpenCL_Level0_gpu = nullptr;

    TestDPPLSyclDeviceInterface ()
    {
        if(DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_CPU)) {
            auto Q = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_CPU, 0);
            OpenCL_cpu = DPPLQueue_GetDevice(Q);
            DPPLQueue_Delete(Q);
        }

        if(DPPLQueueMgr_GetNumQueues(DPPL_OPENCL, DPPL_GPU)) {
            auto Q = DPPLQueueMgr_GetQueue(DPPL_OPENCL, DPPL_GPU, 0);
            OpenCL_gpu = DPPLQueue_GetDevice(Q);
            DPPLQueue_Delete(Q);
        }

        if(DPPLQueueMgr_GetNumQueues(DPPL_LEVEL_ZERO, DPPL_GPU)) {
            auto Q = DPPLQueueMgr_GetQueue(DPPL_LEVEL_ZERO, DPPL_GPU, 0);
            OpenCL_Level0_gpu = DPPLQueue_GetDevice(Q);
            DPPLQueue_Delete(Q);
        }
    }

    ~TestDPPLSyclDeviceInterface ()
    {
        DPPLDevice_Delete(OpenCL_cpu);
        DPPLDevice_Delete(OpenCL_gpu);
        DPPLDevice_Delete(OpenCL_Level0_gpu);
    }

};

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetDriverInfo)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DriverInfo = DPPLDevice_GetDriverInfo(OpenCL_cpu);
    EXPECT_TRUE(DriverInfo != nullptr);
    DPPLCString_Delete(DriverInfo);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetDriverInfo)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DriverInfo = DPPLDevice_GetDriverInfo(OpenCL_gpu);
    EXPECT_TRUE(DriverInfo != nullptr);
    DPPLCString_Delete(DriverInfo);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetDriverInfo)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto DriverInfo = DPPLDevice_GetDriverInfo(OpenCL_Level0_gpu);
    EXPECT_TRUE(DriverInfo != nullptr);
    DPPLCString_Delete(DriverInfo);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetMaxComputeUnits)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPPLDevice_GetMaxComputeUnits(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetMaxComputeUnits)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPPLDevice_GetMaxComputeUnits(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetMaxComputeUnits)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPPLDevice_GetMaxComputeUnits(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetMaxWorkItemDims)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPPLDevice_GetMaxWorkItemDims(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetMaxWorkItemDims)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPPLDevice_GetMaxWorkItemDims(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetMaxWorkItemDims)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPPLDevice_GetMaxWorkItemDims(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetMaxWorkItemSizes)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto item_sizes = DPPLDevice_GetMaxWorkItemSizes(OpenCL_cpu);
    EXPECT_TRUE(item_sizes != nullptr);
    DPPLSize_t_Array_Delete(item_sizes);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetMaxWorkItemSizes)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto item_sizes = DPPLDevice_GetMaxWorkItemSizes(OpenCL_gpu);
    EXPECT_TRUE(item_sizes != nullptr);
    DPPLSize_t_Array_Delete(item_sizes);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetMaxWorkItemSizes)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto item_sizes = DPPLDevice_GetMaxWorkItemSizes(OpenCL_Level0_gpu);
    EXPECT_TRUE(item_sizes != nullptr);
    DPPLSize_t_Array_Delete(item_sizes);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetMaxWorkGroupSize)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPPLDevice_GetMaxWorkGroupSize(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetMaxWorkGroupSize)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPPLDevice_GetMaxWorkGroupSize(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetMaxWorkGroupSize)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPPLDevice_GetMaxWorkGroupSize(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetMaxNumSubGroups)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto n = DPPLDevice_GetMaxNumSubGroups(OpenCL_cpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetMaxNumSubGroups)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto n = DPPLDevice_GetMaxNumSubGroups(OpenCL_gpu);
    EXPECT_TRUE(n > 0);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetMaxNumSubGroups)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto n = DPPLDevice_GetMaxNumSubGroups(OpenCL_Level0_gpu);
    EXPECT_TRUE(n > 0);
}

//TODO: Update when DPC++ properly supports aspects
TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_HasInt64BaseAtomics)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto atomics = DPPLDevice_HasInt64BaseAtomics(OpenCL_cpu);
    auto D = reinterpret_cast<device*>(OpenCL_cpu);
    auto has_atomics= D->has(aspect::int64_base_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

//TODO: Update when DPC++ properly supports aspects
TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_HasInt64BaseAtomics)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto atomics = DPPLDevice_HasInt64BaseAtomics(OpenCL_gpu);
    auto D = reinterpret_cast<device*>(OpenCL_gpu);
    auto has_atomics= D->has(aspect::int64_base_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

//TODO: Update when DPC++ properly supports aspects
TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_HasInt64BaseAtomics)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto atomics = DPPLDevice_HasInt64BaseAtomics(OpenCL_Level0_gpu);
    auto D = reinterpret_cast<device*>(OpenCL_Level0_gpu);
    auto has_atomics= D->has(aspect::int64_base_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

//TODO: Update when DPC++ properly supports aspects
TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_HasInt64ExtendedAtomics)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto atomics = DPPLDevice_HasInt64ExtendedAtomics(OpenCL_cpu);
    auto D = reinterpret_cast<device*>(OpenCL_cpu);
    auto has_atomics= D->has(aspect::int64_extended_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

//TODO: Update when DPC++ properly supports aspects
TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_HasInt64ExtendedAtomics)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.");

    auto atomics = DPPLDevice_HasInt64ExtendedAtomics(OpenCL_gpu);
    auto D = reinterpret_cast<device*>(OpenCL_gpu);
    auto has_atomics= D->has(aspect::int64_extended_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

//TODO: Update when DPC++ properly supports aspects
TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_HasInt64ExtendedAtomics)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto atomics = DPPLDevice_HasInt64ExtendedAtomics(OpenCL_Level0_gpu);
    auto D = reinterpret_cast<device*>(OpenCL_Level0_gpu);
    auto has_atomics= D->has(aspect::int64_extended_atomics);
    EXPECT_TRUE(has_atomics == atomics);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetName)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DevName = DPPLDevice_GetName(OpenCL_cpu);
    EXPECT_TRUE(DevName != nullptr);
    DPPLCString_Delete(DevName);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetName)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto DevName = DPPLDevice_GetName(OpenCL_gpu);
    EXPECT_TRUE(DevName != nullptr);
    DPPLCString_Delete(DevName);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetName)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto DevName = DPPLDevice_GetName(OpenCL_Level0_gpu);
    EXPECT_TRUE(DevName != nullptr);
    DPPLCString_Delete(DevName);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_GetVendorName)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto VendorName = DPPLDevice_GetVendorName(OpenCL_cpu);
    EXPECT_TRUE(VendorName != nullptr);
    DPPLCString_Delete(VendorName);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_GetVendorName)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    auto VendorName = DPPLDevice_GetVendorName(OpenCL_gpu);
    EXPECT_TRUE(VendorName != nullptr);
    DPPLCString_Delete(VendorName);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_GetVendorName)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    auto VendorName = DPPLDevice_GetVendorName(OpenCL_Level0_gpu);
    EXPECT_TRUE(VendorName != nullptr);
    DPPLCString_Delete(VendorName);
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLCPU_IsCPU)
{
    if(!OpenCL_cpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    EXPECT_TRUE(DPPLDevice_IsCPU(OpenCL_cpu));
}

TEST_F (TestDPPLSyclDeviceInterface, CheckOCLGPU_IsGPU)
{
    if(!OpenCL_gpu)
        GTEST_SKIP_("Skipping as no OpenCL CPU device found.");

    EXPECT_TRUE(DPPLDevice_IsGPU(OpenCL_gpu));
}

TEST_F (TestDPPLSyclDeviceInterface, CheckLevel0GPU_IsGPU)
{
    if(!OpenCL_Level0_gpu)
        GTEST_SKIP_("Skipping as no Level0 GPU device found.");

    EXPECT_TRUE(DPPLDevice_IsGPU(OpenCL_Level0_gpu));
}

int
main (int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
