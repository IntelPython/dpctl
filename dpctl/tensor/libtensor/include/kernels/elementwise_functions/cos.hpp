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
namespace cos
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

template <typename argT,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
struct CosContigFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    CosContigFunctor(const argT *inp, resT *res, const size_t nelems)
        : in(inp), out(res), nelems_(nelems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        auto sg = ndit.get_sub_group();

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<argT>::value) {
            std::uint8_t sgSize = sg.get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(base + sgSize * (n_vecs * vec_sz), nelems_);
                 offset += sgSize)
            {
                using realT = typename argT::value_type;
                // cos(x + I*y) = cos(x)*cosh(y) - I*sin(x)*sinh(y)
                auto v = std::real(in[offset]);
                realT cosX_val;
                const realT sinX_val = sycl::sincos(-v, &cosX_val);
                v = std::imag(in[offset]);
                const realT sinhY_val = sycl::sinh(v);
                const realT coshY_val = sycl::cosh(v);

                const realT res_re = coshY_val * cosX_val;
                const realT res_im = sinX_val * sinhY_val;
                out[offset] = resT{res_re, res_im};
            }
        }
        else {
            using dpctl::tensor::type_utils::vec_cast;

            std::uint8_t sgSize = sg.get_local_range()[0];
            size_t base = n_vecs * vec_sz *
                          (ndit.get_group(0) * ndit.get_local_range(0) +
                           sg.get_group_id()[0] * sgSize);
            if (base + n_vecs * vec_sz * sg.get_max_local_range()[0] < nelems_)
            {
                using in_ptrT =
                    sycl::multi_ptr<const argT,
                                    sycl::access::address_space::global_space>;
                using out_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;

#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    sycl::vec<argT, vec_sz> x =
                        sg.load<vec_sz>(in_ptrT(&in[base + it * sgSize]));

                    sycl::vec<resT, vec_sz> res_vec = sycl::cos(
                        vec_cast<resT, typename decltype(x)::element_type,
                                 vec_sz>(x));
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize)
                    out[k] = sycl::cos(static_cast<resT>(in[k]));
            }
        }
    }
};

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

template <typename argT, typename resT, typename IndexerT>
struct CosStridedFunctor
{
private:
    const argT *in = nullptr;
    resT *out = nullptr;
    IndexerT inp_out_indexer_;

public:
    CosStridedFunctor(const argT *inp_tp,
                      resT *res_tp,
                      IndexerT arg_res_indexer)
        : in(inp_tp), out(res_tp), inp_out_indexer_(arg_res_indexer)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        auto offsets_ = inp_out_indexer_(static_cast<py::ssize_t>(wid.get(0)));
        const py::ssize_t &inp_offset = offsets_.get_first_offset();
        const py::ssize_t &out_offset = offsets_.get_second_offset();

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<argT>::value) {
            using realT = typename argT::value_type;
            // cos(x + I*y) = cos(x)*cosh(y) - I*sin(x)*sinh(y)
            auto v = std::real(in[inp_offset]);
            realT cosX_val;
            const realT sinX_val = sycl::sincos(-v, &cosX_val);
            v = std::imag(in[inp_offset]);
            const realT sinhY_val = sycl::sinh(v);
            const realT coshY_val = sycl::cosh(v);

            const realT res_re = coshY_val * cosX_val;
            const realT res_im = sinX_val * sinhY_val;
            out[out_offset] = resT{res_re, res_im};
        }
        else {
            out[out_offset] = std::cos(static_cast<resT>(in[inp_offset]));
        }
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

        cgh.parallel_for<cos_strided_kernel<argTy, resTy, IndexerT>>(
            {nelems}, CosStridedFunctor<argTy, resTy, IndexerT>(
                          arg_tp, res_tp, arg_res_indexer));
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
