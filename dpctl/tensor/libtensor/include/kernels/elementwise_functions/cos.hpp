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

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace cos
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct CosFunctor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in)
    {
        if constexpr (is_complex<argT>::value) {
            using realT = typename argT::value_type;
            // cos(x + I*y) = cos(x)*cosh(y) - I*sin(x)*sinh(y)
            auto v = std::real(in);
            realT cosX_val;
            const realT sinX_val = sycl::sincos(-v, &cosX_val);
            v = std::imag(in);
            const realT sinhY_val = sycl::sinh(v);
            const realT coshY_val = sycl::cosh(v);

            const realT res_re = coshY_val * cosX_val;
            const realT res_im = sinX_val * sinhY_val;
            return resT{res_re, res_im};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return std::cos(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using CosContigFunctor = elementwise_common::
    UnaryContigFunctor<argTy, resTy, CosFunctor<argTy, resTy>, vec_sz, n_vecs>;

template <typename argTy, typename resTy, typename IndexerT>
using CosStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, CosFunctor<argTy, resTy>>;

template <typename T> struct CosOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapEntry<T, float, float>,
        td_ns::TypeMapEntry<T, double, double>,
        td_ns::TypeMapEntry<T, std::complex<float>, std::complex<float>>,
        td_ns::TypeMapEntry<T, std::complex<double>, std::complex<double>>,
        td_ns::DefaultEntry<void>>::result_type;
};

typedef sycl::event (*cos_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class cos_contig_kernel;

template <typename argTy>
sycl::event cos_contig_impl(sycl::queue exec_q,
                            size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event cos_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        constexpr size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        static_assert(lws % vec_sz == 0);
        auto gws_range = sycl::range<1>(
            ((nelems + n_vecs * lws * vec_sz - 1) / (lws * n_vecs * vec_sz)) *
            lws);
        auto lws_range = sycl::range<1>(lws);

        using resTy = typename CosOutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<class cos_contig_kernel<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            CosContigFunctor<argTy, resTy, vec_sz, n_vecs>(arg_tp, res_tp,
                                                           nelems));
    });
    return cos_ev;
}

template <typename fnT, typename T> struct CosContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename CosOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cos_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct CosTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::cos(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename CosOutputType<T>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class cos_strided_kernel;

typedef sycl::event (*cos_strided_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    const py::ssize_t *,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename argTy>
sycl::event cos_strided_impl(sycl::queue exec_q,
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

        using resTy = typename CosOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT arg_res_indexer(nd, arg_offset, res_offset, shape_and_strides);

        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        sycl::range<1> gRange{nelems};

        cgh.parallel_for<cos_strided_kernel<argTy, resTy, IndexerT>>(
            gRange, CosStridedFunctor<argTy, resTy, IndexerT>(arg_tp, res_tp,
                                                              arg_res_indexer));
    });
    return comp_ev;
}

template <typename fnT, typename T> struct CosStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename CosOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cos_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace cos
} // namespace kernels
} // namespace tensor
} // namespace dpctl
