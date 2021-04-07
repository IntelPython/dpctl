//===-- test_sycl_program_interface.cpp - Test cases for module interface -===//
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
/// dpctl_sycl_module_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Config/dpctl_config.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_manager.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_program_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include <CL/sycl.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace cl::sycl;

struct TestDPCTLSyclProgramInterface
    : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclProgramRef PRef = nullptr;
    std::ifstream spirvFile;
    size_t spirvFileSize;
    std::vector<char> spirvBuffer;

    TestDPCTLSyclProgramInterface()
    {
        auto DS = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DS);
        DPCTLDeviceSelector_Delete(DS);
        CRef = DPCTLDeviceMgr_GetCachedContext(DRef);
        QRef = DPCTLQueue_Create(CRef, DRef, nullptr, DPCTL_DEFAULT_PROPERTY);

        if (DRef) {
            spirvFile.open("./multi_kernel.spv",
                           std::ios::binary | std::ios::ate);
            spirvFileSize = std::filesystem::file_size("./multi_kernel.spv");
            spirvBuffer.reserve(spirvFileSize);
            spirvFile.seekg(0, std::ios::beg);
            spirvFile.read(spirvBuffer.data(), spirvFileSize);
            PRef = DPCTLProgram_CreateFromSpirv(CRef, spirvBuffer.data(),
                                                spirvFileSize, nullptr);
        }
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestDPCTLSyclProgramInterface()
    {
        if (DRef)
            spirvFile.close();
        DPCTLDevice_Delete(DRef);
        DPCTLQueue_Delete(QRef);
        DPCTLContext_Delete(CRef);
        DPCTLProgram_Delete(PRef);
    }
};

TEST_P(TestDPCTLSyclProgramInterface, Chk_CreateFromSpirv)
{

    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "axpy"));
}

TEST_P(TestDPCTLSyclProgramInterface, Chk_GetKernel)
{
    auto AddKernel = DPCTLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");

    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);
    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
}

struct TestOCLProgramFromSource : public ::testing::Test
{
    const char *CLProgramStr = R"CLC(
        kernel void add(global int* a, global int* b, global int* c) {
            size_t index = get_global_id(0);
            c[index] = a[index] + b[index];
        }

        kernel void axpy(global int* a, global int* b, global int* c, int d)
        {
            size_t index = get_global_id(0);
            c[index] = a[index] + d*b[index];
        }
    )CLC";
    const char *CompileOpts = "-cl-fast-relaxed-math";
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclProgramRef PRef = nullptr;

    TestOCLProgramFromSource()
    {
        auto DS = DPCTLFilterSelector_Create("opencl:gpu");
        DRef = DPCTLDevice_CreateFromSelector(DS);
        DPCTLDeviceSelector_Delete(DS);
        CRef = DPCTLDeviceMgr_GetCachedContext(DRef);
        QRef = DPCTLQueue_Create(CRef, DRef, nullptr, DPCTL_DEFAULT_PROPERTY);

        if (DRef)
            PRef = DPCTLProgram_CreateFromOCLSource(CRef, CLProgramStr,
                                                    CompileOpts);
    }

    ~TestOCLProgramFromSource()
    {
        DPCTLDevice_Delete(DRef);
        DPCTLQueue_Delete(QRef);
        DPCTLContext_Delete(CRef);
        DPCTLProgram_Delete(PRef);
    }
};

TEST_F(TestOCLProgramFromSource, CheckCreateFromOCLSource)
{
    if (!DRef)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "add"));
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "axpy"));
}

TEST_F(TestOCLProgramFromSource, CheckGetKernelOCLSource)
{
    if (!DRef)
        GTEST_SKIP_("Skipping as no OpenCL GPU device found.\n");

    auto AddKernel = DPCTLProgram_GetKernel(PRef, "add");
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");
    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);
    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
}

INSTANTIATE_TEST_SUITE_P(ProgramCreationFromSpriv,
                         TestDPCTLSyclProgramInterface,
                         ::testing::Values("opencl",
                                           "opencl:gpu",
                                           "opencl:cpu",
                                           "opencl:gpu:0",
#ifdef DPCTL_ENABLE_LO_PROGRAM_CREATION
                                           "level_zero",
                                           "level_zero:gpu",
#endif
                                           "opencl:cpu:0"));
