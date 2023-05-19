#pragma once
#include <CL/sycl.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace isfinite
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct IsFiniteFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    /*
    std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value
    */
    using is_constant = typename std::disjunction<std::is_same<argT, bool>,
                                                  std::is_integral<argT>>;
    static constexpr resT constant_value = true;
    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            const bool real_isfinite = std::isfinite(std::real(in));
            const bool imag_isfinite = std::isfinite(std::imag(in));
            return (real_isfinite && imag_isfinite);
        }
        else if constexpr (std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value)
        {
            return constant_value;
        }
        else if constexpr (std::is_same_v<argT, sycl::half>) {
            return sycl::isfinite(in);
        }
        else {
            return std::isfinite(in);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in)
    {
        auto const &res_vec = sycl::isfinite(in);

        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;

        return vec_cast<bool, deducedT, vec_sz>(res_vec);
    }
};

template <typename argT,
          typename resT = bool,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using IsFiniteContigFunctor = elementwise_common::
    UnaryContigFunctor<argT, resT, IsFiniteFunctor<argT, resT>, vec_sz, n_vecs>;

template <typename argTy, typename resTy, typename IndexerT>
using IsFiniteStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, IsFiniteFunctor<argTy, resTy>>;

template <typename argTy> struct IsFiniteOutputType
{
    using value_type = bool;
};

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class isfinite_contig_kernel;

template <typename argTy>
sycl::event isfinite_contig_impl(sycl::queue exec_q,
                                 size_t nelems,
                                 const char *arg_p,
                                 char *res_p,
                                 const std::vector<sycl::event> &depends = {})
{
    sycl::event isfinite_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        constexpr size_t lws = 64;
        constexpr std::uint8_t vec_sz = 4;
        constexpr std::uint8_t n_vecs = 2;
        static_assert(lws % vec_sz == 0);
        size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        auto gws_range = sycl::range<1>(n_groups * lws);
        auto lws_range = sycl::range<1>(lws);

        using resTy = typename IsFiniteOutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<
            class isfinite_contig_kernel<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            IsFiniteContigFunctor<argTy, resTy, vec_sz, n_vecs>(arg_tp, res_tp,
                                                                nelems));
    });
    return isfinite_ev;
}

template <typename fnT, typename T> struct IsFiniteContigFactory
{
    fnT get()
    {
        fnT fn = isfinite_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T> struct IsFiniteTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isfinite(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename IsFiniteOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class isfinite_strided_kernel;

template <typename argTy>
sycl::event
isfinite_strided_impl(sycl::queue exec_q,
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
    sycl::event isfinite_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename IsFiniteOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT arg_res_indexer{nd, arg_offset, res_offset, shape_and_strides};

        const argTy *arg_tptr = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tptr = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<isfinite_strided_kernel<argTy, resTy, IndexerT>>(
            {nelems}, IsFiniteStridedFunctor<argTy, resTy, IndexerT>(
                          arg_tptr, res_tptr, arg_res_indexer));
    });
    return isfinite_ev;
}

template <typename fnT, typename T> struct IsFiniteStridedFactory
{
    fnT get()
    {
        fnT fn = isfinite_strided_impl<T>;
        return fn;
    }
};

} // namespace isfinite
} // namespace kernels
} // namespace tensor
} // namespace dpctl
