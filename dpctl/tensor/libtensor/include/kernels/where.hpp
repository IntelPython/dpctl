//=== where.hpp -  Implementation of where kernels ---*-C++-*--/===//
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
/// This file defines kernels for dpctl.tensor.where.
//===----------------------------------------------------------------------===//

#pragma once
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "utils/offset_utils.hpp"
#include "utils/type_utils.hpp"
#include <CL/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace search
{

namespace py = pybind11;

using namespace dpctl::tensor::offset_utils;

template <typename T, typename condT, typename IndexerT>
class where_strided_kernel;
template <typename T, typename condT, int vec_sz, int n_vecs>
class where_contig_kernel;

template <typename T, typename condT, int vec_sz = 4, int n_vecs = 2>
class WhereContigFunctor
{
private:
    size_t nelems = 0;
    const char *x1_cp = nullptr;
    const char *x2_cp = nullptr;
    char *dst_cp = nullptr;
    const char *cond_cp = nullptr;

public:
    WhereContigFunctor(size_t nelems_,
                       const char *cond_data_p,
                       const char *x1_data_p,
                       const char *x2_data_p,
                       char *dst_data_p)
        : nelems(nelems_), x1_cp(x1_data_p), x2_cp(x2_data_p),
          dst_cp(dst_data_p), cond_cp(cond_data_p)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        const T *x1_data = reinterpret_cast<const T *>(x1_cp);
        const T *x2_data = reinterpret_cast<const T *>(x2_cp);
        T *dst_data = reinterpret_cast<T *>(dst_cp);
        const condT *cond_data = reinterpret_cast<const condT *>(cond_cp);

        using dpctl::tensor::type_utils::convert_impl;

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<condT>::value || is_complex<T>::value) {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
                bool check = convert_impl<bool, condT>(cond_data[offset]);
                dst_data[offset] = check ? x1_data[offset] : x2_data[offset];
            }
        }
        else {
            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * max_sgSize);

            if (base + n_vecs * vec_sz * sgSize < nelems &&
                sgSize == max_sgSize) {
                using dst_ptrT =
                    sycl::multi_ptr<T,
                                    sycl::access::address_space::global_space>;
                using x_ptrT =
                    sycl::multi_ptr<const T,
                                    sycl::access::address_space::global_space>;
                using cond_ptrT =
                    sycl::multi_ptr<const condT,
                                    sycl::access::address_space::global_space>;

                sycl::vec<T, vec_sz> dst_vec;
                sycl::vec<T, vec_sz> x1_vec;
                sycl::vec<T, vec_sz> x2_vec;
                sycl::vec<condT, vec_sz> cond_vec;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    auto idx = base + it * sgSize;
                    x1_vec = sg.load<vec_sz>(x_ptrT(&x1_data[idx]));
                    x2_vec = sg.load<vec_sz>(x_ptrT(&x2_data[idx]));
                    cond_vec = sg.load<vec_sz>(cond_ptrT(&cond_data[idx]));

#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        bool check = convert_impl<bool, condT>(cond_vec[k]);
                        dst_vec[k] = check ? x1_vec[k] : x2_vec[k];
                    }
                    sg.store<vec_sz>(dst_ptrT(&dst_data[idx]), dst_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems;
                     k += sgSize) {
                    bool check = convert_impl<bool, condT>(cond_data[k]);
                    dst_data[k] = check ? x1_data[k] : x2_data[k];
                }
            }
        }
    }
};

typedef sycl::event (*where_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    const char *,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T, typename condT>
sycl::event where_contig_impl(sycl::queue q,
                              size_t nelems,
                              const char *cond_p,
                              const char *x1_p,
                              const char *x2_p,
                              char *dst_p,
                              const std::vector<sycl::event> &depends)
{
    sycl::event where_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        cgh.parallel_for<where_contig_kernel<T, condT, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            WhereContigFunctor<T, condT, vec_sz, n_vecs>(nelems, cond_p, x1_p,
                                                         x2_p, dst_p));
    });

    return where_ev;
}

template <typename T, typename condT, typename IndexerT>
class WhereStridedFunctor
{
private:
    const char *x1_cp = nullptr;
    const char *x2_cp = nullptr;
    char *dst_cp = nullptr;
    const char *cond_cp = nullptr;
    IndexerT indexer;

public:
    WhereStridedFunctor(const char *cond_data_p,
                        const char *x1_data_p,
                        const char *x2_data_p,
                        char *dst_data_p,
                        IndexerT indexer_)
        : x1_cp(x1_data_p), x2_cp(x2_data_p), dst_cp(dst_data_p),
          cond_cp(cond_data_p), indexer(indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const T *x1_data = reinterpret_cast<const T *>(x1_cp);
        const T *x2_data = reinterpret_cast<const T *>(x2_cp);
        T *dst_data = reinterpret_cast<T *>(dst_cp);
        const condT *cond_data = reinterpret_cast<const condT *>(cond_cp);

        size_t gid = id[0];
        auto offsets = indexer(static_cast<py::ssize_t>(gid));

        using dpctl::tensor::type_utils::convert_impl;
        bool check =
            convert_impl<bool, condT>(cond_data[offsets.get_first_offset()]);

        dst_data[gid] = check ? x1_data[offsets.get_second_offset()]
                              : x2_data[offsets.get_third_offset()];
    }
};

typedef sycl::event (*where_strided_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    const char *,
    const char *,
    const char *,
    char *,
    const py::ssize_t *,
    py::ssize_t,
    py::ssize_t,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename T, typename condT>
sycl::event where_strided_impl(sycl::queue q,
                               size_t nelems,
                               int nd,
                               const char *cond_p,
                               const char *x1_p,
                               const char *x2_p,
                               char *dst_p,
                               const py::ssize_t *shape_strides,
                               py::ssize_t x1_offset,
                               py::ssize_t x2_offset,
                               py::ssize_t cond_offset,
                               const std::vector<sycl::event> &depends)
{
    sycl::event where_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        ThreeOffsets_StridedIndexer indexer{nd, cond_offset, x1_offset,
                                            x2_offset, shape_strides};

        cgh.parallel_for<
            where_strided_kernel<T, condT, ThreeOffsets_StridedIndexer>>(
            sycl::range<1>(nelems),
            WhereStridedFunctor<T, condT, ThreeOffsets_StridedIndexer>(
                cond_p, x1_p, x2_p, dst_p, indexer));
    });

    return where_ev;
}

template <typename fnT, typename T, typename condT> struct WhereStridedFactory
{
    fnT get()
    {
        fnT fn = where_strided_impl<T, condT>;
        return fn;
    }
};

template <typename fnT, typename T, typename condT> struct WhereContigFactory
{
    fnT get()
    {
        fnT fn = where_contig_impl<T, condT>;
        return fn;
    }
};

} // namespace search
} // namespace kernels
} // namespace tensor
} // namespace dpctl
