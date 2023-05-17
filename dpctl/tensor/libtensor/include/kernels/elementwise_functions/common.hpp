//=== common.hpp -  -----------------------------------*-C++-*--/===//
//= Implementation of tensor elementwise operation kernels ------===//
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
/// This file defines kernels for elementwise operations over tensor .
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace elementwise_common
{

/*! @brief Functor for unary function evaluation on contiguous array */
template <typename argT,
          typename resT,
          typename UnaryOperatorT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
struct UnaryContigFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    UnaryContigFunctor(const argT *inp, resT *res, const size_t n_elems)
        : in(inp), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        UnaryOperatorT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: vec_sz must divide sg.max_local_range()[0] */

        if constexpr (UnaryOperatorT::is_constant::value) {
            // value of operator is known to be a known constant
            constexpr resT const_val = UnaryOperatorT::constant_value;
            using out_ptrT =
                sycl::multi_ptr<resT,
                                sycl::access::address_space::global_space>;

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);
            if (base + n_vecs * vec_sz * sgSize < nelems_ &&
                max_sgSize == sgSize) {
                sycl::vec<resT, vec_sz> res_vec(const_val);
#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = const_val;
                }
            }
        }
        else if constexpr (UnaryOperatorT::supports_sg_loadstore::value &&
                           UnaryOperatorT::supports_vec::value)
        {
            using in_ptrT =
                sycl::multi_ptr<const argT,
                                sycl::access::address_space::global_space>;
            using out_ptrT =
                sycl::multi_ptr<bool,
                                sycl::access::address_space::global_space>;

            auto sg = ndit.get_sub_group();
            std::uint16_t sgSize = sg.get_local_range()[0];
            std::uint16_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * max_sgSize);
            if (base + n_vecs * vec_sz * sgSize < nelems_ &&
                sgSize == max_sgSize) {
                sycl::vec<argT, vec_sz> x;

#pragma unroll
                for (std::uint16_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    x = sg.load<vec_sz>(in_ptrT(&in[base + it * sgSize]));
                    sycl::vec<resT, vec_sz> res_vec = op(x);
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    // scalar call
                    out[k] = op(in[k]);
                }
            }
        }
        else if constexpr (UnaryOperatorT::supports_sg_loadstore::value &&
                           std::is_same_v<resT, argT>)
        {
            // default: use scalar-value function

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * maxsgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (maxsgSize == sgSize)) {
                using in_ptrT =
                    sycl::multi_ptr<const argT,
                                    sycl::access::address_space::global_space>;
                using out_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;
                sycl::vec<argT, vec_sz> arg_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    arg_vec = sg.load<vec_sz>(in_ptrT(&in[base + it * sgSize]));
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        arg_vec[k] = op(arg_vec[k]);
                    }
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     arg_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in[k]);
                }
            }
        }
        else if constexpr (UnaryOperatorT::supports_sg_loadstore::value) {
            // default: use scalar-value function

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * maxsgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (maxsgSize == sgSize)) {
                using in_ptrT =
                    sycl::multi_ptr<const argT,
                                    sycl::access::address_space::global_space>;
                using out_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;
                sycl::vec<argT, vec_sz> arg_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    arg_vec = sg.load<vec_sz>(in_ptrT(&in[base + it * sgSize]));
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        res_vec[k] = op(arg_vec[k]);
                    }
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in[k]);
                }
            }
        }
        else {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems_, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
                out[offset] = op(in[offset]);
            }
        }
    }
};

template <typename argT, typename resT, typename IndexerT, typename UnaryOpT>
struct UnaryStridedFunctor
{
private:
    const argT *inp_ = nullptr;
    resT *res_ = nullptr;
    IndexerT inp_out_indexer_;

public:
    UnaryStridedFunctor(const argT *inp_p,
                        resT *res_p,
                        IndexerT inp_out_indexer)
        : inp_(inp_p), res_(res_p), inp_out_indexer_(inp_out_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &offsets_ = inp_out_indexer_(wid.get(0));
        const py::ssize_t &inp_offset = offsets_.get_first_offset();
        const py::ssize_t &res_offset = offsets_.get_second_offset();

        UnaryOpT op{};

        res_[res_offset] = op(inp_[inp_offset]);
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename BinaryOperatorT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
struct BinaryContigFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    BinaryContigFunctor(const argT1 *inp1,
                        const argT2 *inp2,
                        resT *res,
                        const size_t n_elems)
        : in1(inp1), in2(inp2), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        BinaryOperatorT op{};
        /* Each work-item processes vec_sz elements, contiguous in memory */

        if constexpr (BinaryOperatorT::supports_sg_loadstore::value &&
                      BinaryOperatorT::supports_vec::value)
        {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];

            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (sgSize == maxsgSize)) {
                using in_ptrT1 =
                    sycl::multi_ptr<const argT1,
                                    sycl::access::address_space::global_space>;
                using in_ptrT2 =
                    sycl::multi_ptr<const argT2,
                                    sycl::access::address_space::global_space>;
                using out_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;
                sycl::vec<argT1, vec_sz> arg1_vec;
                sycl::vec<argT2, vec_sz> arg2_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    arg1_vec =
                        sg.load<vec_sz>(in_ptrT1(&in1[base + it * sgSize]));
                    arg2_vec =
                        sg.load<vec_sz>(in_ptrT2(&in2[base + it * sgSize]));
                    res_vec = op(arg1_vec, arg2_vec);
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in1[k], in2[k]);
                }
            }
        }
        else if constexpr (BinaryOperatorT::supports_sg_loadstore::value) {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t maxsgSize = sg.get_max_local_range()[0];

            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);

            if ((base + n_vecs * vec_sz * sgSize < nelems_) &&
                (sgSize == maxsgSize)) {
                using in_ptrT1 =
                    sycl::multi_ptr<const argT1,
                                    sycl::access::address_space::global_space>;
                using in_ptrT2 =
                    sycl::multi_ptr<const argT2,
                                    sycl::access::address_space::global_space>;
                using out_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;
                sycl::vec<argT1, vec_sz> arg1_vec;
                sycl::vec<argT2, vec_sz> arg2_vec;
                sycl::vec<resT, vec_sz> res_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    arg1_vec =
                        sg.load<vec_sz>(in_ptrT1(&in1[base + it * sgSize]));
                    arg2_vec =
                        sg.load<vec_sz>(in_ptrT2(&in2[base + it * sgSize]));
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        res_vec[vec_id] =
                            op(arg1_vec[vec_id], arg2_vec[vec_id]);
                    }
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = op(in1[k], in2[k]);
                }
            }
        }
        else {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems_, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
                out[offset] = op(in1[offset], in2[offset]);
            }
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename ThreeOffsets_IndexerT,
          typename BinaryOperatorT>
struct BinaryStridedFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    ThreeOffsets_IndexerT three_offsets_indexer_;

public:
    BinaryStridedFunctor(const argT1 *inp1_tp,
                         const argT2 *inp2_tp,
                         resT *res_tp,
                         ThreeOffsets_IndexerT inps_res_indexer)
        : in1(inp1_tp), in2(inp2_tp), out(res_tp),
          three_offsets_indexer_(inps_res_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &three_offsets_ =
            three_offsets_indexer_(static_cast<py::ssize_t>(wid.get(0)));

        const auto &inp1_offset = three_offsets_.get_first_offset();
        const auto &inp2_offset = three_offsets_.get_second_offset();
        const auto &out_offset = three_offsets_.get_third_offset();

        BinaryOperatorT op{};
        out[out_offset] = op(in1[inp1_offset], in2[inp2_offset]);
    }
};

} // namespace elementwise_common
} // namespace kernels
} // namespace tensor
} // namespace dpctl
