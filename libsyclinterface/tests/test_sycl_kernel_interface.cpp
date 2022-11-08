//===-- test_sycl_kernel_interface.cpp - Test cases for kernel interface  ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2022 Intel Corporation
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
/// dpctl_sycl_kernel_interface.h.
///
//===----------------------------------------------------------------------===//

#include "Config/dpctl_config.h"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_kernel_bundle_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_queue_manager.h"
#include "dpctl_utils.h"
#include <CL/sycl.hpp>
#include <array>
#include <gtest/gtest.h>

using namespace sycl;

namespace
{
struct TestDPCTLSyclKernelInterface
    : public ::testing::TestWithParam<const char *>
{
    const char *CLProgramStr = R"CLC(
        kernel void add(global int* a, global int* b, global int* c) {
            size_t index = get_global_id(0);
            c[index] = a[index] + b[index];
        }

        kernel void axpy(global int* a, global int* b, global int* c, int d) {
            size_t index = get_global_id(0);
            c[index] = a[index] + d*b[index];
        }
    )CLC";
    const char *CompileOpts = "-cl-fast-relaxed-math";
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclContextRef CtxRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;
    DPCTLSyclKernelRef AddKRef = nullptr;
    DPCTLSyclKernelRef AxpyKRef = nullptr;

    TestDPCTLSyclKernelInterface()
    {
        DSRef = DPCTLFilterSelector_Create(GetParam());
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
        QRef = DPCTLQueue_CreateForDevice(DRef, nullptr, 0);
        CtxRef = DPCTLQueue_GetContext(QRef);
        KBRef = DPCTLKernelBundle_CreateFromOCLSource(
            CtxRef, DRef, CLProgramStr, CompileOpts);
        AddKRef = DPCTLKernelBundle_GetKernel(KBRef, "add");
        AxpyKRef = DPCTLKernelBundle_GetKernel(KBRef, "axpy");
    }

    ~TestDPCTLSyclKernelInterface()
    {
        DPCTLDeviceSelector_Delete(DSRef);
        DPCTLDevice_Delete(DRef);
        DPCTLQueue_Delete(QRef);
        DPCTLContext_Delete(CtxRef);
        DPCTLKernelBundle_Delete(KBRef);
        DPCTLKernel_Delete(AddKRef);
        DPCTLKernel_Delete(AxpyKRef);
    }

    void SetUp()
    {
        if (!DRef) {
            auto message = "Skipping as no device of type " +
                           std::string(GetParam()) + ".";
            GTEST_SKIP_(message.c_str());
        }
    }
};
} // namespace

TEST_P(TestDPCTLSyclKernelInterface, CheckGetNumArgs)
{
    ASSERT_EQ(DPCTLKernel_GetNumArgs(AddKRef), 3ul);
    ASSERT_EQ(DPCTLKernel_GetNumArgs(AxpyKRef), 4ul);
}

TEST_P(TestDPCTLSyclKernelInterface, CheckCopy)
{
    DPCTLSyclKernelRef Copied_KRef = nullptr;
    EXPECT_NO_FATAL_FAILURE(Copied_KRef = DPCTLKernel_Copy(AddKRef));
    ASSERT_EQ(DPCTLKernel_GetNumArgs(Copied_KRef), 3ul);
    EXPECT_NO_FATAL_FAILURE(DPCTLKernel_Delete(Copied_KRef));
}

TEST_P(TestDPCTLSyclKernelInterface, CheckGetWorkGroupSize)
{

    size_t add_wgsz = 0, axpy_wgsz = 0;
    EXPECT_NO_FATAL_FAILURE(add_wgsz = DPCTLKernel_GetWorkGroupSize(AddKRef));
    EXPECT_NO_FATAL_FAILURE(axpy_wgsz = DPCTLKernel_GetWorkGroupSize(AxpyKRef));

    ASSERT_TRUE(add_wgsz != 0);
    ASSERT_TRUE(axpy_wgsz != 0);
}

TEST_P(TestDPCTLSyclKernelInterface, CheckGetPreferredWorkGroupSizeMultiple)
{

    size_t add_wgsz_m = 0, axpy_wgsz_m = 0;
    EXPECT_NO_FATAL_FAILURE(
        add_wgsz_m = DPCTLKernel_GetPreferredWorkGroupSizeMultiple(AddKRef));
    EXPECT_NO_FATAL_FAILURE(
        axpy_wgsz_m = DPCTLKernel_GetPreferredWorkGroupSizeMultiple(AxpyKRef));

    ASSERT_TRUE(add_wgsz_m != 0);
    ASSERT_TRUE(axpy_wgsz_m != 0);
}

TEST_P(TestDPCTLSyclKernelInterface, CheckGetPrivateMemSize)
{

    size_t add_private_mem_sz = 0, axpy_private_mem_sz = 0;
    EXPECT_NO_FATAL_FAILURE(add_private_mem_sz =
                                DPCTLKernel_GetPrivateMemSize(AddKRef));
    EXPECT_NO_FATAL_FAILURE(axpy_private_mem_sz =
                                DPCTLKernel_GetPrivateMemSize(AxpyKRef));

    if (DPCTLDevice_IsGPU(DRef)) {
        ASSERT_TRUE(add_private_mem_sz != 0);
        ASSERT_TRUE(axpy_private_mem_sz != 0);
    }
    else {
        ASSERT_TRUE(add_private_mem_sz >= 0);
        ASSERT_TRUE(axpy_private_mem_sz >= 0);
    }
}

TEST_P(TestDPCTLSyclKernelInterface, CheckGetMaxNumSubGroups)
{

    uint32_t add_mnsg = 0, axpy_mnsg = 0;
    EXPECT_NO_FATAL_FAILURE(add_mnsg = DPCTLKernel_GetMaxNumSubGroups(AddKRef));
    EXPECT_NO_FATAL_FAILURE(axpy_mnsg =
                                DPCTLKernel_GetMaxNumSubGroups(AxpyKRef));

    ASSERT_TRUE(add_mnsg != 0);
    ASSERT_TRUE(axpy_mnsg != 0);
}

#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_2023_SWITCHOVER
TEST_P(TestDPCTLSyclKernelInterface, CheckGetMaxSubGroupSize)
{

    uint32_t add_msg_sz = 0, axpy_msg_sz = 0;
    EXPECT_NO_FATAL_FAILURE(add_msg_sz =
                                DPCTLKernel_GetMaxSubGroupSize(AddKRef));
    EXPECT_NO_FATAL_FAILURE(axpy_msg_sz =
                                DPCTLKernel_GetMaxSubGroupSize(AxpyKRef));

    ASSERT_TRUE(add_msg_sz != 0);
    ASSERT_TRUE(axpy_msg_sz != 0);
}
#endif

TEST_P(TestDPCTLSyclKernelInterface, CheckGetCompileNumSubGroups)
{

    uint32_t add_cnsg = 0, axpy_cnsg = 0;
    EXPECT_NO_FATAL_FAILURE(add_cnsg =
                                DPCTLKernel_GetCompileNumSubGroups(AddKRef));
    EXPECT_NO_FATAL_FAILURE(axpy_cnsg =
                                DPCTLKernel_GetCompileNumSubGroups(AxpyKRef));

    EXPECT_TRUE(add_cnsg >= 0);
    EXPECT_TRUE(axpy_cnsg >= 0);
}

TEST_P(TestDPCTLSyclKernelInterface, CheckGetCompileSubGroupSize)
{

    uint32_t add_csg_sz = 0, axpy_csg_sz = 0;
    EXPECT_NO_FATAL_FAILURE(add_csg_sz =
                                DPCTLKernel_GetCompileSubGroupSize(AddKRef));
    EXPECT_NO_FATAL_FAILURE(axpy_csg_sz =
                                DPCTLKernel_GetCompileSubGroupSize(AxpyKRef));
    EXPECT_TRUE(add_csg_sz >= 0);
    EXPECT_TRUE(axpy_csg_sz >= 0);
}

INSTANTIATE_TEST_SUITE_P(TestKernelInterfaceFunctions,
                         TestDPCTLSyclKernelInterface,
                         ::testing::Values("opencl:gpu:0", "opencl:cpu:0"));

struct TestDPCTLSyclKernelNullArgs : public ::testing::Test
{
    DPCTLSyclKernelRef Null_KRef;
    TestDPCTLSyclKernelNullArgs() : Null_KRef(nullptr) {}
    ~TestDPCTLSyclKernelNullArgs() {}
};

TEST_F(TestDPCTLSyclKernelNullArgs, CheckNumArgsNullKRef)
{
    ASSERT_EQ(DPCTLKernel_GetNumArgs(Null_KRef), -1);
}

TEST_F(TestDPCTLSyclKernelNullArgs, CheckCopyNullKRef)
{
    ASSERT_TRUE(DPCTLKernel_Copy(Null_KRef) == nullptr);
}

TEST_F(TestDPCTLSyclKernelNullArgs, CheckGetWorkGroupsSizeNullKRef)
{
    DPCTLSyclKernelRef NullKRef = nullptr;

    ASSERT_EQ(DPCTLKernel_GetWorkGroupSize(NullKRef), 0);
}

TEST_F(TestDPCTLSyclKernelNullArgs,
       CheckGetPreferredWorkGroupsSizeMultipleNullKRef)
{
    DPCTLSyclKernelRef NullKRef = nullptr;

    ASSERT_EQ(DPCTLKernel_GetPreferredWorkGroupSizeMultiple(NullKRef), 0);
}

TEST_F(TestDPCTLSyclKernelNullArgs, CheckGetPrivateMemSizeNullKRef)
{
    DPCTLSyclKernelRef NullKRef = nullptr;

    ASSERT_EQ(DPCTLKernel_GetPrivateMemSize(NullKRef), 0);
}

TEST_F(TestDPCTLSyclKernelNullArgs, CheckGetMaxNumSubGroupsNullKRef)
{
    DPCTLSyclKernelRef NullKRef = nullptr;

    ASSERT_EQ(DPCTLKernel_GetMaxNumSubGroups(NullKRef), 0);
}

#if __SYCL_COMPILER_VERSION >= __SYCL_COMPILER_2023_SWITCHOVER
TEST_F(TestDPCTLSyclKernelNullArgs, CheckGetMaxSubGroupSizeNullKRef)
{
    DPCTLSyclKernelRef NullKRef = nullptr;

    ASSERT_EQ(DPCTLKernel_GetMaxSubGroupSize(NullKRef), 0);
}
#endif

TEST_F(TestDPCTLSyclKernelNullArgs, CheckGetCompileNumSubGroupsNullKRef)
{
    DPCTLSyclKernelRef NullKRef = nullptr;

    ASSERT_EQ(DPCTLKernel_GetCompileNumSubGroups(NullKRef), 0);
}

TEST_F(TestDPCTLSyclKernelNullArgs, CheckGetCompileSubGroupSizeNullKRef)
{
    DPCTLSyclKernelRef NullKRef = nullptr;

    ASSERT_EQ(DPCTLKernel_GetCompileSubGroupSize(NullKRef), 0);
}
