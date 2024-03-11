//===-- test_sycl_queue_submit.cpp - Test cases for kernel submission fns. ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
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

#include "dpctl_sycl_context_interface.h"
#include "dpctl_sycl_device_interface.h"
#include "dpctl_sycl_device_selector_interface.h"
#include "dpctl_sycl_event_interface.h"
#include "dpctl_sycl_kernel_bundle_interface.h"
#include "dpctl_sycl_kernel_interface.h"
#include "dpctl_sycl_queue_interface.h"
#include "dpctl_sycl_type_casters.hpp"
#include "dpctl_sycl_usm_interface.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sycl/sycl.hpp>
#include <utility>

namespace
{
constexpr size_t SIZE = 1024;
static_assert(SIZE % 8 == 0);

using namespace dpctl::syclinterface;

template <typename T>
void submit_kernel(DPCTLSyclQueueRef QRef,
                   DPCTLSyclKernelBundleRef KBRef,
                   std::vector<char> spirvBuffer,
                   size_t spirvFileSize,
                   DPCTLKernelArgType kernelArgTy,
                   std::string kernelName)
{
    T scalarVal = 3;
    constexpr size_t NARGS = 4;
    constexpr size_t RANGE_NDIMS_1 = 1;
    constexpr size_t RANGE_NDIMS_2 = 2;
    constexpr size_t RANGE_NDIMS_3 = 3;

    ASSERT_TRUE(DPCTLKernelBundle_HasKernel(KBRef, kernelName.c_str()));
    auto kernel = DPCTLKernelBundle_GetKernel(KBRef, kernelName.c_str());

    // Create the input args
    auto a = DPCTLmalloc_shared(SIZE * sizeof(T), QRef);
    ASSERT_TRUE(a != nullptr);
    auto b = DPCTLmalloc_shared(SIZE * sizeof(T), QRef);
    ASSERT_TRUE(b != nullptr);
    auto c = DPCTLmalloc_shared(SIZE * sizeof(T), QRef);
    ASSERT_TRUE(c != nullptr);

    // Create kernel args for vector_add
    size_t Range[] = {SIZE};
    void *args[NARGS] = {unwrap<void>(a), unwrap<void>(b), unwrap<void>(c),
                         (void *)&scalarVal};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                              DPCTL_VOID_PTR, kernelArgTy};
    auto E1Ref =
        DPCTLQueue_SubmitRange(kernel, QRef, args, addKernelArgTypes, NARGS,
                               Range, RANGE_NDIMS_1, nullptr, 0);
    ASSERT_TRUE(E1Ref != nullptr);

    // Create kernel args for vector_add
    size_t Range2D[] = {SIZE, 1};
    DPCTLSyclEventRef DepEvs[] = {E1Ref};
    auto E2Ref =
        DPCTLQueue_SubmitRange(kernel, QRef, args, addKernelArgTypes, NARGS,
                               Range2D, RANGE_NDIMS_2, DepEvs, 1);
    ASSERT_TRUE(E2Ref != nullptr);

    // Create kernel args for vector_add
    size_t Range3D[] = {SIZE, 1, 1};
    DPCTLSyclEventRef DepEvs2[] = {E1Ref, E2Ref};
    auto E3Ref =
        DPCTLQueue_SubmitRange(kernel, QRef, args, addKernelArgTypes, NARGS,
                               Range3D, RANGE_NDIMS_3, DepEvs2, 2);
    ASSERT_TRUE(E3Ref != nullptr);

    DPCTLEvent_Wait(E3Ref);

    // clean ups
    DPCTLEvent_Delete(E1Ref);
    DPCTLEvent_Delete(E2Ref);
    DPCTLEvent_Delete(E3Ref);
    DPCTLKernel_Delete(kernel);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)a, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)b, QRef);
    DPCTLfree_with_queue((DPCTLSyclUSMRef)c, QRef);
}

} /* end of anonymous namespace */

/*
// The oneD_range_kernel spv files were generated from the SYCL program included
// in this comment. The program can be compiled using
// `icpx -fsycl oneD_range_kernel.cpp`. After that if the generated executable
// is run with the environment variable `SYCL_DUMP_IMAGES=1`, icpx runtime
// will dump all offload sections of fat binary to the current working
// directory. When tested with DPC++ 2024.0 the kernels are split across two
// separate SPV files. One contains all kernels for integers and FP32
// data type, and another contains the kernel for FP64.
//
// Note that, `SYCL_DUMP_IMAGES=1` will also generate extra SPV files that
// contain the code for built in functions such as indexing and barriers. To
// figure which SPV file contains the kernels, use `spirv-dis` from the
// spirv-tools package to translate the SPV binary format to a human-readable
// textual format.

#include <CL/sycl.hpp>
#include <iostream>
#include <sstream>

template <typename T>
class Range1DKernel
{
private:
    T *a_ = nullptr;
    T *b_ = nullptr;
    T *c_ = nullptr;
    T scalarVal_;

public:
    RangeKernel(T *a, T *b, T *c, T scalarVal)
        : a_(a), b_(b), c_(c), scalarVal_(scalarVal)
    {
    }

    void operator()(sycl::item<1> it) const
    {
        auto i = it.get_id();
        a_[i] = i + 1;
        b_[i] = i + 2;
        c_[i] = scalarVal_ * (a_[i] + b_[i]);
    }
};

template <typename T>
void submit_kernel(
    sycl::queue q,
    const unsigned long N,
    T *a,
    T *b,
    T *c,
    T scalarVal)
{
    // clang-format off
    q.submit([&](auto &h) {
        h.parallel_for(sycl::range(N), RangeKernel<T>(a, b, c, scalarVal));
    });
    // clang-format on
}

template <typename T>
void driver(size_t N)
{
    sycl::queue q;
    auto *a = sycl::malloc_shared<T>(N, q);
    auto *b = sycl::malloc_shared<T>(N, q);
    auto *c = sycl::malloc_shared<T>(N, q);
    T scalarVal = 3;

    submit_kernel(q, N, a, b, c, scalarVal);
    q.wait();

    std::cout << "C[0] : " << (size_t)c[0] << " " << std::endl;
    sycl::free(a, q);
}

int main(int argc, const char **argv)
{
    size_t N = 0;
    std::cout << "Enter problem size in N:\n";
    std::cin >> N;
    std::cout << "Executing with N = " << N << std::endl;

    driver<int8_t>(N);
    driver<uint8_t>(N);
    driver<int16_t>(N);
    driver<uint16_t>(N);
    driver<int32_t>(N);
    driver<uint32_t>(N);
    driver<int64_t>(N);
    driver<uint64_t>(N);
    driver<float>(N);
    driver<double>(N);

    return 0;
}
*/

struct TestQueueSubmit : public ::testing::Test
{
    std::ifstream spirvFile;
    size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmit()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        DPCTLSyclDeviceRef DRef = nullptr;

        spirvFile.open("./oneD_range_kernel_inttys_fp32.spv",
                       std::ios::binary | std::ios::ate);
        spirvFileSize_ =
            std::filesystem::file_size("./oneD_range_kernel_inttys_fp32.spv");
        spirvBuffer_.reserve(spirvFileSize_);
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer_.data(), spirvFileSize_);

        DSRef = DPCTLDefaultSelector_Create();
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
        QRef =
            DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
        auto CRef = DPCTLQueue_GetContext(QRef);

        KBRef = DPCTLKernelBundle_CreateFromSpirv(
            CRef, DRef, spirvBuffer_.data(), spirvFileSize_, nullptr);
        DPCTLDevice_Delete(DRef);
        DPCTLDeviceSelector_Delete(DSRef);
    }

    ~TestQueueSubmit()
    {
        spirvFile.close();
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

struct TestQueueSubmitFP64 : public ::testing::Test
{
    std::ifstream spirvFile;
    size_t spirvFileSize_;
    std::vector<char> spirvBuffer_;
    DPCTLSyclDeviceRef DRef = nullptr;
    DPCTLSyclQueueRef QRef = nullptr;
    DPCTLSyclKernelBundleRef KBRef = nullptr;

    TestQueueSubmitFP64()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;

        spirvFile.open("./oneD_range_kernel_fp64.spv",
                       std::ios::binary | std::ios::ate);
        spirvFileSize_ =
            std::filesystem::file_size("./oneD_range_kernel_fp64.spv");
        spirvBuffer_.reserve(spirvFileSize_);
        spirvFile.seekg(0, std::ios::beg);
        spirvFile.read(spirvBuffer_.data(), spirvFileSize_);
        DSRef = DPCTLDefaultSelector_Create();
        DRef = DPCTLDevice_CreateFromSelector(DSRef);
        QRef =
            DPCTLQueue_CreateForDevice(DRef, nullptr, DPCTL_DEFAULT_PROPERTY);
        auto CRef = DPCTLQueue_GetContext(QRef);

        KBRef = DPCTLKernelBundle_CreateFromSpirv(
            CRef, DRef, spirvBuffer_.data(), spirvFileSize_, nullptr);
        DPCTLDeviceSelector_Delete(DSRef);
    }

    ~TestQueueSubmitFP64()
    {
        spirvFile.close();
        DPCTLDevice_Delete(DRef);
        DPCTLQueue_Delete(QRef);
        DPCTLKernelBundle_Delete(KBRef);
    }
};

TEST_F(TestQueueSubmit, CheckForInt8)
{
    submit_kernel<int8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                          DPCTLKernelArgType::DPCTL_INT8_T,
                          "_ZTS11RangeKernelIaE");
}

TEST_F(TestQueueSubmit, CheckForUInt8)
{
    submit_kernel<uint8_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_UINT8_T,
                           "_ZTS11RangeKernelIhE");
}

TEST_F(TestQueueSubmit, CheckForInt16)
{
    submit_kernel<int16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_INT16_T,
                           "_ZTS11RangeKernelIsE");
}

TEST_F(TestQueueSubmit, CheckForUInt16)
{
    submit_kernel<uint16_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                            DPCTLKernelArgType::DPCTL_UINT16_T,
                            "_ZTS11RangeKernelItE");
}

TEST_F(TestQueueSubmit, CheckForInt32)
{
    submit_kernel<int32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_INT32_T,
                           "_ZTS11RangeKernelIiE");
}

TEST_F(TestQueueSubmit, CheckForUInt32)
{
    submit_kernel<uint32_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                            DPCTLKernelArgType::DPCTL_UINT32_T,
                            "_ZTS11RangeKernelIjE");
}

TEST_F(TestQueueSubmit, CheckForInt64)
{
    submit_kernel<int64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                           DPCTLKernelArgType::DPCTL_INT64_T,
                           "_ZTS11RangeKernelIlE");
}

TEST_F(TestQueueSubmit, CheckForUInt64)
{
    submit_kernel<uint64_t>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                            DPCTLKernelArgType::DPCTL_UINT64_T,
                            "_ZTS11RangeKernelImE");
}

TEST_F(TestQueueSubmit, CheckForFloat)
{
    submit_kernel<float>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                         DPCTLKernelArgType::DPCTL_FLOAT32_T,
                         "_ZTS11RangeKernelIfE");
}

TEST_F(TestQueueSubmitFP64, CheckForDouble)
{
    if (DPCTLDevice_HasAspect(DRef, DPCTLSyclAspectType::fp64)) {
        submit_kernel<double>(QRef, KBRef, spirvBuffer_, spirvFileSize_,
                              DPCTLKernelArgType::DPCTL_FLOAT64_T,
                              "_ZTS11RangeKernelIdE");
    }
}

TEST_F(TestQueueSubmit, CheckForUnsupportedArgTy)
{

    int scalarVal = 3;
    size_t Range[] = {SIZE};
    size_t RANGE_NDIMS = 1;
    constexpr size_t NARGS = 4;

    auto kernel = DPCTLKernelBundle_GetKernel(KBRef, "_ZTS11RangeKernelIdE");
    void *args[NARGS] = {unwrap<void>(nullptr), unwrap<void>(nullptr),
                         unwrap<void>(nullptr), (void *)&scalarVal};
    DPCTLKernelArgType addKernelArgTypes[] = {DPCTL_VOID_PTR, DPCTL_VOID_PTR,
                                              DPCTL_VOID_PTR,
                                              DPCTL_UNSUPPORTED_KERNEL_ARG};
    auto ERef = DPCTLQueue_SubmitRange(kernel, QRef, args, addKernelArgTypes,
                                       NARGS, Range, RANGE_NDIMS, nullptr, 0);

    ASSERT_TRUE(ERef == nullptr);
}

struct TestQueueSubmitBarrier : public ::testing::Test
{
    DPCTLSyclQueueRef QRef = nullptr;

    TestQueueSubmitBarrier()
    {
        DPCTLSyclDeviceSelectorRef DSRef = nullptr;
        DPCTLSyclDeviceRef DRef = nullptr;

        EXPECT_NO_FATAL_FAILURE(DSRef = DPCTLDefaultSelector_Create());
        EXPECT_NO_FATAL_FAILURE(DRef = DPCTLDevice_CreateFromSelector(DSRef));
        EXPECT_NO_FATAL_FAILURE(QRef = DPCTLQueue_CreateForDevice(
                                    DRef, nullptr, DPCTL_DEFAULT_PROPERTY));
        EXPECT_NO_FATAL_FAILURE(DPCTLDevice_Delete(DRef));
        EXPECT_NO_FATAL_FAILURE(DPCTLDeviceSelector_Delete(DSRef));
    }
    ~TestQueueSubmitBarrier()
    {
        EXPECT_NO_FATAL_FAILURE(DPCTLQueue_Delete(QRef));
    }
};

TEST_F(TestQueueSubmitBarrier, ChkSubmitBarrier)
{
    DPCTLSyclEventRef ERef = nullptr;

    ASSERT_TRUE(QRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(ERef = DPCTLQueue_SubmitBarrier(QRef));
    ASSERT_TRUE(ERef != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));
}

TEST_F(TestQueueSubmitBarrier, ChkSubmitBarrierWithEvents)
{
    DPCTLSyclEventRef ERef = nullptr;
    DPCTLSyclEventRef DepsERefs[2] = {nullptr, nullptr};

    EXPECT_NO_FATAL_FAILURE(DepsERefs[0] = DPCTLEvent_Create());
    EXPECT_NO_FATAL_FAILURE(DepsERefs[1] = DPCTLEvent_Create());

    ASSERT_TRUE(QRef != nullptr);
    EXPECT_NO_FATAL_FAILURE(
        ERef = DPCTLQueue_SubmitBarrierForEvents(QRef, DepsERefs, 2));

    ASSERT_TRUE(ERef != nullptr);
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Wait(ERef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(ERef));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(DepsERefs[0]));
    EXPECT_NO_FATAL_FAILURE(DPCTLEvent_Delete(DepsERefs[1]));
}
