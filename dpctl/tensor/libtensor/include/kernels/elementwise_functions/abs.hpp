#pragma once
#include <CL/sycl.hpp>

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
namespace abs
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
struct AbsContigFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    AbsContigFunctor(const argT *inp, resT *res, const size_t n_elems)
        : in(inp), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: vec_sz must divide sg.max_local_range()[0] */

        if constexpr (std::is_same_v<argT, bool> ||
                      (std::is_integral<argT>::value &&
                       std::is_unsigned<argT>::value))
        {
            static_assert(std::is_same_v<resT, argT>);

            auto sg = ndit.get_sub_group();
            std::uint8_t sgSize = sg.get_local_range()[0];
            std::uint8_t max_sgSize = sg.get_max_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * max_sgSize);

            if (base + n_vecs * vec_sz * sgSize < nelems_ &&
                sgSize == max_sgSize) {
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
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     arg_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = in[k];
                }
            }
        }
        else {
            using dpctl::tensor::type_utils::is_complex;
            if constexpr (is_complex<argT>::value) {
                std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
                size_t base = ndit.get_global_linear_id();

                base = (base / sgSize) * sgSize * n_vecs * vec_sz +
                       (base % sgSize);
                for (size_t offset = base;
                     offset <
                     std::min(nelems_, base + sgSize * (n_vecs * vec_sz));
                     offset += sgSize)
                {
                    out[offset] = std::abs(in[offset]);
                }
            }
            else {
                auto sg = ndit.get_sub_group();
                std::uint8_t sgSize = sg.get_local_range()[0];
                std::uint8_t maxsgSize = sg.get_max_local_range()[0];
                size_t base = n_vecs * vec_sz *
                              (ndit.get_group(0) * ndit.get_local_range(0) +
                               sg.get_group_id()[0] * maxsgSize);

                if (base + n_vecs * vec_sz < nelems_) {
                    using in_ptrT = sycl::multi_ptr<
                        const argT, sycl::access::address_space::global_space>;
                    using out_ptrT = sycl::multi_ptr<
                        resT, sycl::access::address_space::global_space>;
                    sycl::vec<argT, vec_sz> arg_vec;

#pragma unroll
                    for (std::uint8_t it = 0; it < n_vecs * vec_sz;
                         it += vec_sz) {
                        arg_vec =
                            sg.load<vec_sz>(in_ptrT(&in[base + it * sgSize]));
#pragma unroll
                        for (std::uint8_t k = 0; k < vec_sz; ++k) {
                            arg_vec[k] = std::abs(arg_vec[k]);
                        }
                        sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                         arg_vec);
                    }
                }
                else {
                    for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                         k += sgSize) {
                        out[k] = std::abs(in[k]);
                    }
                }
            }
        }
    }
};

template <typename T> struct AbsOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::TypeMapEntry<T, bool>,
        td_ns::TypeMapEntry<T, std::uint8_t>,
        td_ns::TypeMapEntry<T, std::uint16_t>,
        td_ns::TypeMapEntry<T, std::uint32_t>,
        td_ns::TypeMapEntry<T, std::uint64_t>,
        td_ns::TypeMapEntry<T, std::int8_t>,
        td_ns::TypeMapEntry<T, std::int16_t>,
        td_ns::TypeMapEntry<T, std::int32_t>,
        td_ns::TypeMapEntry<T, std::int64_t>,
        td_ns::TypeMapEntry<T, sycl::half>,
        td_ns::TypeMapEntry<T, float>,
        td_ns::TypeMapEntry<T, double>,
        td_ns::TypeMapEntry<T, std::complex<float>, float>,
        td_ns::TypeMapEntry<T, std::complex<double>, double>,
        td_ns::DefaultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class abs_contig_kernel;

typedef sycl::event (*abs_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    char *,
    const std::vector<sycl::event> &);

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
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argT, typename resT, typename IndexerT>
struct AbsStridedFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    IndexerT inp_res_indexer_;

public:
    AbsStridedFunctor(const argT *inp_p,
                      resT *res_p,
                      IndexerT two_offsets_indexer)
        : in(inp_p), out(res_p), inp_res_indexer_(two_offsets_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        auto offsets_ = inp_res_indexer_(static_cast<py::ssize_t>(wid[0]));
        const auto &inp_offset = offsets_.get_first_offset();
        const auto &out_offset = offsets_.get_second_offset();

        if constexpr (std::is_same_v<argT, bool> ||
                      (std::is_integral<argT>::value &&
                       std::is_unsigned<argT>::value))
        {
            out[out_offset] = in[inp_offset];
        }
        else {
            out[out_offset] = std::abs(in[inp_offset]);
        }
    }
};

template <typename T1, typename T2, typename T3> class abs_strided_kernel;

typedef sycl::event (*abs_strided_impl_fn_ptr_t)(
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
    sycl::event abs_ev = exec_q.submit([&](sycl::handler &cgh) {
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
    return abs_ev;
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
