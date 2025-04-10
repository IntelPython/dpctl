//===- test_sycl_kernel_bundle_interface.cpp -
//                                      Test cases for module interface -===//
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

#include <stddef.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>

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

TEST_P(TestDPCTLSyclKernelBundleInterface, ChkCopy)
{
    DPCTLSyclKernelBundleRef Copied_KBRef = nullptr;
    ASSERT_TRUE(KBRef != nullptr);

    EXPECT_NO_FATAL_FAILURE(Copied_KBRef = DPCTLKernelBundle_Copy(KBRef));
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(Copied_KBRef, "add"));
    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(Copied_KBRef, "axpy"));

    EXPECT_NO_FATAL_FAILURE(DPCTLKernelBundle_Delete(Copied_KBRef));
}

TEST_P(TestDPCTLSyclKernelBundleInterface, ChkCopyNullArgument)
{
    DPCTLSyclKernelBundleRef Null_KBRef = nullptr;
    DPCTLSyclKernelBundleRef Copied_KBRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(Copied_KBRef = DPCTLKernelBundle_Copy(Null_KBRef));
    ASSERT_TRUE(Copied_KBRef == nullptr);
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

struct TestSYCLKernelBundleFromSource
    : public ::testing::TestWithParam<const char *>
{
    const char *sycl_source = R"===(
    #include <sycl/sycl.hpp>
    #include "math_ops.hpp"
    #include "math_template_ops.hpp"

    namespace syclext = sycl::ext::oneapi::experimental;

    extern "C" SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclext::nd_range_kernel<1>))
    void vector_add(int* in1, int* in2, int* out){
        sycl::nd_item<1> item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
        size_t globalID = item.get_global_linear_id();
        out[globalID] = math_op(in1[globalID],in2[globalID]);
    }

    template<typename T>
    SYCL_EXTERNAL SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclext::nd_range_kernel<1>))
    void vector_add_template(T* in1, T* in2, T* out){
        sycl::nd_item<1> item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
        size_t globalID = item.get_global_linear_id();
        out[globalID] = math_op_template(in1[globalID], in2[globalID]);
    }
    )===";

    const char *header1_content = R"===(
    int math_op(int a, int b){
        return a + b;
    }
    )===";

    const char *header2_content = R"===(
    template<typename T>
    T math_op_template(T a, T b){
        return a + b;
    }
    )===";

    const char *CompileOpt = "-fno-fast-math";
    const char *KernelName = "vector_add_template<int>";
    const char *Header1Name = "math_ops.hpp";
    const char *Header2Name = "math_template_ops.hpp";
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclContextRef CRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestSYCLKernelBundleFromSource()
    {
        auto DS = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DS);
        DPCTLDeviceSelector_Delete(DS);
        CRef = DPCTLDeviceMgr_GetCachedContext(DRef);

        if (DRef) {
            DPCTLBuildOptionListRef BORef = DPCTLBuildOptionList_Create();
            DPCTLBuildOptionList_Append(BORef, CompileOpt);
            DPCTLKernelNameListRef KNRef = DPCTLKernelNameList_Create();
            DPCTLKernelNameList_Append(KNRef, KernelName);
            DPCTLVirtualHeaderListRef VHRef = DPCTLVirtualHeaderList_Create();
            DPCTLVirtualHeaderList_Append(VHRef, Header1Name, header1_content);
            DPCTLVirtualHeaderList_Append(VHRef, Header2Name, header2_content);
            DPCTLKernelBuildLogRef KBLRef = DPCTLKernelBuildLog_Create();
            KBRef = DPCTLKernelBundle_CreateFromSYCLSource(
                CRef, DRef, sycl_source, VHRef, KNRef, BORef, KBLRef);
            DPCTLVirtualHeaderList_Delete(VHRef);
            DPCTLKernelNameList_Delete(KNRef);
            DPCTLBuildOptionList_Delete(BORef);
            DPCTLKernelBuildLog_Delete(KBLRef);
        }
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
        if (!DPCTLDevice_CanCompileSYCL(DRef)) {
            const char *message = "Skipping as SYCL compilation not supported";
            GTEST_SKIP_(message);
        }
    }

    ~TestSYCLKernelBundleFromSource()
    {
        if (DRef)
            DPCTLDevice_Delete(DRef);
        if (CRef)
            DPCTLContext_Delete(CRef);
        if (KBRef)
            DPCTLKernelBundle_Delete(KBRef);
    }
};

TEST_P(TestSYCLKernelBundleFromSource, CheckCreateFromSYCLSource)
{
    ASSERT_TRUE(KBRef != nullptr);
    ASSERT_TRUE(DPCTLKernelBundle_HasSyclKernel(KBRef, "vector_add"));
    // DPC++ version 2025.1 supports compilation of SYCL template kernels,
    // but does not yet support referencing them with the unmangled name.
    ASSERT_TRUE(
        DPCTLKernelBundle_HasSyclKernel(KBRef, "vector_add_template<int>") ||
        DPCTLKernelBundle_HasSyclKernel(
            KBRef, "_Z33__sycl_kernel_vector_add_templateIiEvPT_S1_S1_"));
}

TEST_P(TestSYCLKernelBundleFromSource, CheckGetKernelSYCLSource)
{
    auto AddKernel = DPCTLKernelBundle_GetSyclKernel(KBRef, "vector_add");
    auto AxpyKernel =
        DPCTLKernelBundle_GetSyclKernel(KBRef, "vector_add_template<int>");
    if (AxpyKernel == nullptr) {
        // DPC++ version 2025.1 supports compilation of SYCL template kernels,
        // but does not yet support referencing them with the unmangled name.
        AxpyKernel = DPCTLKernelBundle_GetSyclKernel(
            KBRef, "_Z33__sycl_kernel_vector_add_templateIiEvPT_S1_S1_");
    }

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

INSTANTIATE_TEST_SUITE_P(KernelBundleCreationFromSYCL,
                         TestSYCLKernelBundleFromSource,
                         ::testing::Values("opencl:gpu",
                                           "opencl:cpu",
                                           "level_zero:gpu"));

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
