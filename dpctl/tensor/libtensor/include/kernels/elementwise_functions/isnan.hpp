#pragma once
#include <CL/sycl.hpp>
#include <cstdint>

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
namespace isnan
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

template <typename argT,
          typename resT,
          std::uint8_t vec_sz = 4,
          std::uint8_t n_vecs = 2>
struct IsNanContigFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    IsNanContigFunctor(const argT *inp, resT *res, const size_t nelems)
        : in(inp), out(res), nelems_(nelems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        using dpctl::tensor::type_utils::is_complex;
        using dpctl::tensor::type_utils::vec_cast;

        if constexpr (is_complex<argT>::value) {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems_, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
                const bool real_isnan = sycl::isnan(std::real(in[offset]));
                const bool imag_isnan = sycl::isnan(std::imag(in[offset]));
                out[offset] = real_isnan || imag_isnan;
            }
        }
        else if constexpr (std::is_same<argT, bool>::value ||
                           std::is_integral<argT>::value)
        {
            using out_ptrT =
                sycl::multi_ptr<resT,
                                sycl::access::address_space::global_space>;

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);
            if (base + n_vecs * vec_sz * max_sgSize < nelems_ &&
                max_sgSize == sgSize) {
                sycl::vec<bool, vec_sz> res_vec(false);
#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = false;
                }
            }
        }
        else {
            using in_ptrT =
                sycl::multi_ptr<const argT,
                                sycl::access::address_space::global_space>;
            using out_ptrT =
                sycl::multi_ptr<bool,
                                sycl::access::address_space::global_space>;
            static_assert(std::is_same<resT, bool>::value);

            auto sg = ndit.get_sub_group();
            std::uint16_t sgSize = sg.get_local_range()[0];
            std::uint16_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * max_sgSize);
            if (base + n_vecs * vec_sz * max_sgSize < nelems_ &&
                sgSize == max_sgSize) {
                sycl::vec<argT, vec_sz> x;

#pragma unroll
                for (std::uint16_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    x = sg.load<vec_sz>(in_ptrT(&in[base + it * sgSize]));
                    // returns vec<int, vec_sz>
                    auto res_vec = sycl::isnan(x);
                    // cast it to bool
                    sycl::vec<bool, vec_sz> res_bool =
                        vec_cast<bool, typename decltype(res_vec)::element_type,
                                 vec_sz>(res_vec);
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_bool);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = static_cast<bool>(sycl::isnan(in[k]));
                }
            }
        }
    }
};

template <typename argTy> struct IsNanOutputType
{
    using value_type = bool;
};

typedef sycl::event (*isnan_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class isnan_contig_kernel;

template <typename argTy>
sycl::event isnan_contig_impl(sycl::queue exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res_p,
                              const std::vector<sycl::event> &depends = {})
{
    sycl::event isnan_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        constexpr size_t lws = 64;
        constexpr std::uint8_t vec_sz = 4;
        constexpr std::uint8_t n_vecs = 2;
        static_assert(lws % vec_sz == 0);
        auto gws_range = sycl::range<1>(
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz)) *
            lws);
        auto lws_range = sycl::range<1>(lws);

        using resTy = typename IsNanOutputType<argTy>::value_type;
        const argTy *arg_tp = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<
            class isnan_contig_kernel<argTy, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            IsNanContigFunctor<argTy, resTy, vec_sz, n_vecs>(arg_tp, res_tp,
                                                             nelems));
    });
    return isnan_ev;
}

template <typename fnT, typename T> struct IsNanContigFactory
{
    fnT get()
    {
        fnT fn = isnan_contig_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T> struct IsNanTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isnan(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename IsNanOutputType<T>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argT, typename resT, typename IndexerT>
struct IsNanStridedFunctor
{
private:
    const argT *inp_ = nullptr;
    resT *res_ = nullptr;
    IndexerT inp_out_indexer_;

public:
    IsNanStridedFunctor(const argT *inp_p,
                        resT *res_p,
                        IndexerT inp_out_indexer)
        : inp_(inp_p), res_(res_p), inp_out_indexer_(inp_out_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const argT *const &in = inp_;
        resT *const &out = res_;

        auto offsets_ = inp_out_indexer_(wid.get(0));
        const py::ssize_t &inp_offset = offsets_.get_first_offset();
        const py::ssize_t &out_offset = offsets_.get_second_offset();

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (std::is_same_v<argT, bool> ||
                      (std::is_integral<argT>::value)) {
            out[out_offset] = false;
        }
        else if constexpr (is_complex<argT>::value) {
            const bool real_isnan = sycl::isnan(std::real(in[inp_offset]));
            const bool imag_isnan = sycl::isnan(std::imag(in[inp_offset]));

            out[out_offset] = real_isnan || imag_isnan;
        }
        else {
            out[out_offset] = sycl::isnan(in[inp_offset]);
        }
    }
};

template <typename T1, typename T2, typename T3> class isnan_strided_kernel;

typedef sycl::event (*isnan_strided_impl_fn_ptr_t)(
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
sycl::event
isnan_strided_impl(sycl::queue exec_q,
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
    sycl::event abs_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename IsNanOutputType<argTy>::value_type;
        using IndexerT =
            typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;

        IndexerT arg_res_indexer{nd, arg_offset, res_offset, shape_and_strides};

        const argTy *arg_tptr = reinterpret_cast<const argTy *>(arg_p);
        resTy *res_tptr = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<isnan_strided_kernel<argTy, resTy, IndexerT>>(
            {nelems}, IsNanStridedFunctor<argTy, resTy, IndexerT>(
                          arg_tptr, res_tptr, arg_res_indexer));
    });
    return abs_ev;
}

template <typename fnT, typename T> struct IsNanStridedFactory
{
    fnT get()
    {
        fnT fn = isnan_strided_impl<T>;
        return fn;
    }
};

} // namespace isnan
} // namespace kernels
} // namespace tensor
} // namespace dpctl
