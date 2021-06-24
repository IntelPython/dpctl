//===-- test_sycl_queue_submit.cpp - Test cases for kernel submission fns. ===//
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
/// This file has unit test cases for the various submit functions defined
/// inside dpctl_sycl_queue_interface.cpp.
//===----------------------------------------------------------------------===//

#include "Support/CBindingWrapping.h"
#include "dpcpp_kernels.hpp"
#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_event_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_program_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_usm_interface.h"
#include <CL/sycl.hpp>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace
{
constexpr size_t SIZE = 1024;
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(void, DPCTLSyclUSMRef);
} /* end of anonymous namespace */

struct TestQueueSubmit : public ::testing::Test
{
    std::ifstream spirvFile;
    size_t spirvFileSize;
    std::vector<char> spirvBuffer;

    TestQueueSubmit()
    {
        spirvFile.open("./multi_kernel.spv", std::ios::binary | std::ios::ate);
        spirvFileSize = std::filesystem::file_size("./multi_kernel.spv");
        spirvBuffer.reserve(spirvFileSize);
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer.data(), spirvFileSize);
    }

    ~TestQueueSubmit()
    {
        spirvFile.close();
    }
};

TEST_F(TestQueueSubmit, CheckSubmitRange_saxpy)
{
    DPCTLSyclDeviceSelectorRef DSRef = nullptr;
    DPCTLSyclDeviceRef DRef = nullptr;

    EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
    EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
    DPCTLDeviceMgr_PrintDeviceInfo(DRef);
    ASSERT_TRUE(DRef);
    auto QRef =
        DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
    ASSERT_TRUE(QRef);
    auto CRef = DPCTLQueue_GetContext(QRef);
    ASSERT_TRUE(CRef);
    auto PRef = DPCTLProgram_CreateFromSpirv(CRef, spirvBuffer.data(),
                                             spirvFileSize, nullptr);
    ASSERT_TRUE(PRef != nullptr);
    ASSERT_TRUE(DPCTLProgram_HasKernel(PRef, "axpy"));
    auto AxpyKernel = DPCTLProgram_GetKernel(PRef, "axpy");

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(a != nullptr);
    auto b = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(b != nullptr);
    auto c = DPCTLmalloc_shared(SIZE * sizeof(float), QRef);
    ASSERT_TRUE(c != nullptr);

    auto a_ptr = reinterpret_cast<float *>(unwrap(a));
    auto b_ptr = reinterpret_cast<float *>(unwrap(b));
    // Initialize a,b
    for (auto i = 0ul; i < SIZE; ++i) {
        a_ptr[i] = i + 1.0;
        b_ptr[i] = i + 2.0;
    }

    // Create kernel args for axpy
    float d = 10.0;
    size_t Range[] = {SIZE};
    void *args2[4] = {unwrap(a), unwrap(b), unwrap(c), (void *)&d};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                              DPCTL_VOID_PTR, DPCTL_FLOAT};
    auto ERef = DPCTLQueue_SubmitRange(
        AxpyKernel, QRef, args2, addKernelArgTypes, 4, Range, 1, nullptr, 0);
    ASSERT_TRUE(ERef != nullptr);
    DPCTLQueue_Wait(QRef);

    // clean ups
    DPCTLEvent_Delete(ERef);
    DPCTLKernel_Delete(AxpyKernel);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)b, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)c, QRef);
    DPCTLQueue_Delete(QRef);
    DPCTLContext_Delete(CRef);
    DPCTLProgram_Delete(PRef);
    DPCTLDevice_Delete(DRef);
    DPCTLDeviceSelector_Delete(DSRef);
}

#ifndef DPCTL_COVERAGE
namespace
{

template <typename T,
          DPCTLKernelArgType katT,
          typename scT,
          DPCTLKernelArgType katscT>
bool common_submit_range_fn(sycl::queue &q, size_t n, scT val)
{
    T *a = sycl::malloc_device<T>(n, q);
    T *b = sycl::malloc_device<T>(n, q);
    T *c = sycl::malloc_device<T>(n, q);
    T fill_val = 1;
    size_t Range[] = {n};

    auto popA_kernel = dpcpp_kernels::get_fill_kernel<T>(q, n, a, fill_val);
    auto popB_kernel = dpcpp_kernels::get_range_kernel<T>(q, n, b);
    auto mad_kernel = dpcpp_kernels::get_mad_kernel<T, scT>(q, n, a, b, c, val);

    DPCTLSyclKernelRef popAKernRef =
        reinterpret_cast<DPCTLSyclKernelRef>(&popA_kernel);
    DPCTLSyclKernelRef popBKernRef =
        reinterpret_cast<DPCTLSyclKernelRef>(&popB_kernel);
    DPCTLSyclKernelRef madKernRef =
        reinterpret_cast<DPCTLSyclKernelRef>(&mad_kernel);

    DPCTLSyclQueueRef QRef = reinterpret_cast<DPCTLSyclQueueRef>(&q);
    void *popAArgs[] = {reinterpret_cast<void *>(a),
                        reinterpret_cast<void *>(&fill_val)};
    DPCTLKernelArgType popAKernelArgTypes[] = {DPCTL_VOID_PTR, katT};

    DPCTLSyclEventRef popAERef =
        DPCTLQueue_SubmitRange(popAKernRef, QRef, popAArgs, popAKernelArgTypes,
                               2, Range, 1, nullptr, 0);

    void *popBArgs[] = {reinterpret_cast<void *>(b)};
    DPCTLKernelArgType popBKernelArgTypes[] = {DPCTL_VOID_PTR};

    DPCTLSyclEventRef popBERef =
        DPCTLQueue_SubmitRange(popBKernRef, QRef, popBArgs, popBKernelArgTypes,
                               1, Range, 1, nullptr, 0);

    void *madArgs[] = {reinterpret_cast<void *>(a), reinterpret_cast<void *>(b),
                       reinterpret_cast<void *>(c),
                       reinterpret_cast<void *>(&val)};
    DPCTLKernelArgType madKernelArgTypes[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                              DPCTL_VOID_PTR, katscT};

    DPCTLSyclEventRef deps[2] = {popAERef, popBERef};
    DPCTLSyclEventRef madRef = DPCTLQueue_SubmitRange(
        madKernRef, QRef, madArgs, madKernelArgTypes, 4, Range, 1, deps, 2);

    DPCTLQueue_Wait(QRef);
    DPCTLEvent_Delete(madRef);
    DPCTLEvent_Delete(popBERef);
    DPCTLEvent_Delete(popAERef);

    bool worked = true;
    T *host_data = new T[n];
    q.memcpy(host_data, c, n * sizeof(T)).wait();
    for (size_t i = 0; i < n; ++i) {
        worked = worked && (host_data[i] == T(fill_val) + T(i) * T(val));
    }
    delete[] host_data;

    sycl::free(c, q);
    sycl::free(b, q);
    sycl::free(a, q);

    return worked;
}

} // namespace

struct TestQueueSubmitRange : public ::testing::Test
{
    sycl::queue q;
    size_t n_elems = 512;

    TestQueueSubmitRange() : q(sycl::default_selector{}) {}
    ~TestQueueSubmitRange() {}
};

TEST_F(TestQueueSubmitRange, ChkSubmitRangeInt)
{
    bool worked = false;
    worked = common_submit_range_fn<int, DPCTL_INT, int, DPCTL_INT>(q, n_elems,
                                                                    int(-1));
    EXPECT_TRUE(worked);
}
#endif
