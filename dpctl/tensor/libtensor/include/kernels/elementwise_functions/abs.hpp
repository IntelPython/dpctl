#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "kernels/elementwise_functions/common.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"
#include <pybind11/pybind11.h>

#include <iostream>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace abs
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct AbsFunctor
{

    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &x)
    {

        if constexpr (std::is_same_v<argT, bool> ||
                      (std::is_integral<argT>::value &&
                       std::is_unsigned<argT>::value))
        {
            static_assert(std::is_same_v<resT, argT>);
            return x;
        }
        else {
            return std::abs(x);
        }
    }
};

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using AbsContigFunctor = elementwise_common::
    UnaryContigFunctor<argT, resT, AbsFunctor<argT, resT>, vec_sz, n_vecs>;

template <typename T> struct AbsOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapResultEntry<T, bool>,
        td_ns::TypeMapResultEntry<T, std::uint8_t>,
        td_ns::TypeMapResultEntry<T, std::uint16_t>,
        td_ns::TypeMapResultEntry<T, std::uint32_t>,
        td_ns::TypeMapResultEntry<T, std::uint64_t>,
        td_ns::TypeMapResultEntry<T, std::int8_t>,
        td_ns::TypeMapResultEntry<T, std::int16_t>,
        td_ns::TypeMapResultEntry<T, std::int32_t>,
        td_ns::TypeMapResultEntry<T, std::int64_t>,
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>, float>,
        td_ns::TypeMapResultEntry<T, std::complex<double>, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class abs_contig_kernel;

template <typename argTy>
sycl::event abs_contig_impl(sycl::queue exec_q,
                            size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event abs_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        using resTy = typename AbsOutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<abs_contig_kernel<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            AbsContigFunctor<argTy, resTy, vec_sz, n_vecs>(arg_tp, res_tp,
                                                           nelems));
    });
    return abs_ev;
}

template <typename fnT, typename T> struct AbsContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AbsOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = abs_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct AbsTypeMapFactory
{
    /*! @brief get typeid for output type of std::abs(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AbsOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argTy, typename resTy, typename IndexerT>
using AbsStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, AbsFunctor<argTy, resTy>>;

template <typename T1, typename T2, typename T3> class abs_strided_kernel;

template <typename argTy>
sycl::event abs_strided_impl(sycl::queue exec_q,
                             size_t nelems,
                             int nd,
                             const py::ssize_t *shape_and_strides,
                             const char *arg_p,
                             py::ssize_t arg_offset,
                             char *res_p,
                             py::ssize_t res_offset,
                             const std::vector<sycl::event> &depends,
                             const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename AbsOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT indexer{nd, arg_offset, res_offset, shape_and_strides};

        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<abs_strided_kernel<argTy, resTy, IndexerT>>(
            {nelems},
            AbsStridedFunctor<argTy, resTy, IndexerT>(arg_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename fnT, typename T> struct AbsStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AbsOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = abs_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace abs
} // namespace kernels
} // namespace tensor
} // namespace dpctl
