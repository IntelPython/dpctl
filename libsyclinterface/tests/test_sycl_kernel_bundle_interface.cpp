//===- test_sycl_kernel_bundle_interface.cpp -
//                                      Test cases for module interface -===//
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
#include "dpctl_sycl_kernel_bundle_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include <CL/sycl.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace sycl;

struct TestDPCTLSyclKernelBundleInterface
    : public ::testing::TestWithParam<const char *>
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;
    std::ifstream spirvFile;
    size_t spirvFileSize;
    std::vector<char> spirvBuffer;

    TestDPCTLSyclKernelBundleInterface()
    {
        auto DS = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DS);
        DPCTLDeviceSelector_Delete(DS);
        CRef = DPCTLDeviceMgr_GetCachedContext(DRef);

        if (DRef) {
            spirvFile.open("./multi_kernel.spv",
                           std::ios::binary | std::ios::ate);
            spirvFileSize = std::filesystem::file_size("./multi_kernel.spv");
            spirvBuffer.reserve(spirvFileSize);
            spirvFile.seekg(0, std::ios::beg);
            spirvFile.read(spirvBuffer.data(), spirvFileSize);
            KBRef = DPCTLKernelBundle_CreateFromSpirv(
                CRef, DRef, spirvBuffer.data(), spirvFileSize, nullptr);
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

    ~TestDPCTLSyclKernelBundleInterface()
    {
        if (DRef) {
            spirvFile.close();
            DPCTLDevice_Delete(DRef);
        }
        if (CRef)
            DPCTLContext_Delete(CRef);
        if (KBRef)
            DPCTLKernelBundle_Delete(KBRef);
    }
};

TEST_P(TestDPCTLSyclKernelBundleInterface, ChkCreateFromSpirv)
{

    ASSERT_TRUE(KBRef != nullptr);
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, "add"));
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, "axpy"));
    ASSERT_FALSE(DPCTLKernelBundle_HasKernel(KBRef, nullptr));
}

TEST_P(TestDPCTLSyclKernelBundleInterface, ChkCreateFromSpirvNull)
{
    DPCTLSyclContextRef Null_CRef = nullptr;
    DPCTLSyclDeviceRef Null_DRef = nullptr;
    const void *null_spirv = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;
    // Null context
    EXPECT_NO_FATAL_FAILURE(KBRef = DPCTLKernelBundle_CreateFromSpirv(
                                Null_CRef, Null_DRef, null_spirv, 0, nullptr));
    ASSERT_TRUE(KBRef == nullptr);

    // Null device
    EXPECT_NO_FATAL_FAILURE(KBRef = DPCTLKernelBundle_CreateFromSpirv(
                                CRef, Null_DRef, null_spirv, 0, nullptr));
    ASSERT_TRUE(KBRef == nullptr);

    // Null IL
    EXPECT_NO_FATAL_FAILURE(KBRef = DPCTLKernelBundle_CreateFromSpirv(
                                CRef, DRef, null_spirv, 0, nullptr));
    ASSERT_TRUE(KBRef == nullptr);
}

TEST_P(TestDPCTLSyclKernelBundleInterface, ChkHasKernelNullProgram)
{

    DPCTLSyclKernelBundleRef NullRef = nullptr;
    ASSERT_FALSE(DPCTLKernelBundle_HasKernel(NullRef, "add"));
}

TEST_P(TestDPCTLSyclKernelBundleInterface, ChkGetKernel)
{
    auto AddKernel = DPCTLKernelBundle_GetKernel(KBRef, "add");
    auto AxpyKernel = DPCTLKernelBundle_GetKernel(KBRef, "axpy");
    auto NullKernel = DPCTLKernelBundle_GetKernel(KBRef, nullptr);

    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);
    ASSERT_TRUE(NullKernel == nullptr);
    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
    EXPECT_NO_FATAL_FAILURE(DPCTLKernel_Delete(NullKernel));
}

TEST_P(TestDPCTLSyclKernelBundleInterface, ChkGetKernelNullProgram)
{
    DPCTLSyclKernelBundleRef NullRef = nullptr;
    DPCTLSyclKernelRef KRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(KRef = DPCTLKernelBundle_GetKernel(NullRef, "add"));
    EXPECT_TRUE(KRef == nullptr);
}

struct TestOCLKernelBundleFromSource
    : public ::testing::TestWithParam<const char *>
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
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestOCLKernelBundleFromSource()
    {
        auto DS = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DS);
        DPCTLDeviceSelector_Delete(DS);
        CRef = DPCTLDeviceMgr_GetCachedContext(DRef);

        if (DRef)
            KBRef = DPCTLKernelBundle_CreateFromOCLSource(
                CRef, DRef, CLProgramStr, CompileOpts);
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestOCLKernelBundleFromSource()
    {
        if (DRef)
            DPCTLDevice_Delete(DRef);
        if (CRef)
            DPCTLContext_Delete(CRef);
        if (KBRef)
            DPCTLKernelBundle_Delete(KBRef);
    }
};

TEST_P(TestOCLKernelBundleFromSource, CheckCreateFromOCLSource)
{
    ASSERT_TRUE(KBRef != nullptr);
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, "add"));
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, "axpy"));
}

TEST_P(TestOCLKernelBundleFromSource, CheckCreateFromOCLSourceNull)
{
    const char *InvalidCLProgramStr = R"CLC(
        kernel void invalid(global foo* a, global bar* b) {
            size_t index = get_global_id(0);
            b[index] = a[index];
        }
    )CLC";
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(KBRef = DPCTLKernelBundle_CreateFromOCLSource(
                                CRef, DRef, InvalidCLProgramStr, CompileOpts););
    ASSERT_TRUE(KBRef == nullptr);

    DPCTLSyclContextRef Null_CRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(
        KBRef = DPCTLKernelBundle_CreateFromOCLSource(
            Null_CRef, DRef, InvalidCLProgramStr, CompileOpts););
    ASSERT_TRUE(KBRef == nullptr);

    DPCTLSyclDeviceRef Null_DRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(
        KBRef = DPCTLKernelBundle_CreateFromOCLSource(
            CRef, Null_DRef, InvalidCLProgramStr, CompileOpts););
    ASSERT_TRUE(KBRef == nullptr);
}

TEST_P(TestOCLKernelBundleFromSource, CheckGetKernelOCLSource)
{
    auto AddKernel = DPCTLKernelBundle_GetKernel(KBRef, "add");
    auto AxpyKernel = DPCTLKernelBundle_GetKernel(KBRef, "axpy");
    ASSERT_TRUE(AddKernel != nullptr);
    ASSERT_TRUE(AxpyKernel != nullptr);
    DPCTLKernel_Delete(AddKernel);
    DPCTLKernel_Delete(AxpyKernel);
}

INSTANTIATE_TEST_SUITE_P(KernelBundleCreationFromSpirv,
                         TestDPCTLSyclKernelBundleInterface,
                         ::testing::Values("opencl",
                                           "opencl:gpu",
                                           "opencl:cpu",
                                           "opencl:gpu:0",
#ifdef DPCTL_ENABLE_L0_PROGRAM_CREATION
                                           "level_zero",
                                           "level_zero:gpu",
#endif
                                           "opencl:cpu:0"));

INSTANTIATE_TEST_SUITE_P(KernelBundleCreationFromSource,
                         TestOCLKernelBundleFromSource,
                         ::testing::Values("opencl:gpu", "opencl:cpu"));

struct TestKernelBundleUnsupportedBackend : public ::testing::Test
{
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclContextRef CRef = nullptr;

    TestKernelBundleUnsupportedBackend()
    {
        auto DS = DPCTLFilterSelector_Create("host:host");
        DRef = DPCTLDevice_CreateFromSelector(DS);
        DPCTLDeviceSelector_Delete(DS);
        if (DRef)
            CRef = DPCTLDeviceMgr_GetCachedContext(DRef);
    }

    void SetUp()
    {
        if (!DRef) {
            std::string message = "Skipping as host device is not enabled.";
            GTEST_SKIP_(message.c_str());
        }
    }

    ~TestKernelBundleUnsupportedBackend()
    {
        if (DRef)
            DPCTLDevice_Delete(DRef);
        if (CRef)
            DPCTLContext_Delete(CRef);
    }
};

TEST_F(TestKernelBundleUnsupportedBackend, CheckCreateFromSource)
{
    const char *src = R"CLC(
        kernel void set(global int* a, int v) {
            size_t index = get_global_id(0);
            a[index] = v;
        }
    )CLC";
    const char *opts = "";

    DPCTLSyclKernelBundleRef KBRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(
        KBRef = DPCTLKernelBundle_CreateFromOCLSource(CRef, DRef, src, opts));
    ASSERT_TRUE(KBRef == nullptr);
}

TEST_F(TestKernelBundleUnsupportedBackend, CheckCreateFromSpirv)
{
    std::ifstream spirvFile;
    size_t spirvFileSize;
    std::vector<char> spirvBuffer;

    spirvFile.open("./multi_kernel.spv", std::ios::binary | std::ios::ate);
    spirvFileSize = std::filesystem::file_size("./multi_kernel.spv");
    spirvBuffer.reserve(spirvFileSize);
    spirvFile.seekg(0, std::ios::beg);
    spirvFile.read(spirvBuffer.data(), spirvFileSize);
    spirvFile.close();

    DPCTLSyclKernelBundleRef KBRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(
        KBRef = DPCTLKernelBundle_CreateFromSpirv(
            CRef, DRef, spirvBuffer.data(), spirvFileSize, nullptr));
    ASSERT_TRUE(KBRef == nullptr);
}
