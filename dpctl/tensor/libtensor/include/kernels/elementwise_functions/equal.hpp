#pragma once
#include <CL/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "kernels/elementwise_functions/common.hpp"
#include <pybind11/pybind11.h>

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace equal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT> struct EqualFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec = std::conjunction<
        std::is_same<argT1, argT2>,
        std::negation<std::disjunction<tu_ns::is_complex<argT1>,
                                       tu_ns::is_complex<argT2>>>>;

    resT operator()(const argT1 &in1, const argT2 &in2)
    {
        return (in1 == in2);
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT1, vec_sz> &in1,
                                       const sycl::vec<argT2, vec_sz> &in2)
    {
        auto tmp = (in1 == in2);
        if constexpr (std::is_same_v<resT,
                                     typename decltype(tmp)::element_type>) {
            return tmp;
        }
        else {
            using dpctl::tensor::type_utils::vec_cast;

            return vec_cast<resT, typename decltype(tmp)::element_type, vec_sz>(
                tmp);
        }
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
using EqualContigFunctor =
    elementwise_common::BinaryContigFunctor<argT1,
                                            argT2,
                                            resT,
                                            EqualFunctor<argT1, argT2, resT>,
                                            vec_sz,
                                            n_vecs>;

template <typename argT1, typename argT2, typename resT, typename IndexerT>
using EqualStridedFunctor =
    elementwise_common::BinaryStridedFunctor<argT1,
                                             argT2,
                                             resT,
                                             IndexerT,
                                             EqualFunctor<argT1, argT2, resT>>;

template <typename T1, typename T2> struct EqualOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::BinaryTypeMapResultEntry<T1, bool, T2, bool, bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::uint8_t, T2, std::uint8_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, std::int8_t, T2, std::int8_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int16_t, T2, std::int16_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int32_t, T2, std::int32_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        bool>,
        td_ns::
            BinaryTypeMapResultEntry<T1, std::int64_t, T2, std::int64_t, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, sycl::half, T2, sycl::half, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, bool>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        bool>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class equal_contig_kernel;

typedef sycl::event (*equal_contig_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    const char *,
    py::ssize_t,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename argTy1, typename argTy2>
sycl::event equal_contig_impl(sycl::queue exec_q,
                              size_t nelems,
                              const char *arg1_p,
                              py::ssize_t arg1_offset,
                              const char *arg2_p,
                              py::ssize_t arg2_offset,
                              char *res_p,
                              py::ssize_t res_offset,
                              const std::vector<sycl::event> &depends = {})
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        using resTy = typename EqualOutputType<argTy1, argTy2>::value_type;

        const argTy1 *arg1_tp =
            reinterpret_cast<const argTy1 *>(arg1_p) + arg1_offset;
        const argTy2 *arg2_tp =
            reinterpret_cast<const argTy2 *>(arg2_p) + arg2_offset;
        resTy *res_tp = reinterpret_cast<resTy *>(res_p) + res_offset;

        cgh.parallel_for<
            equal_contig_kernel<argTy1, argTy2, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            EqualContigFunctor<argTy1, argTy2, resTy, vec_sz, n_vecs>(
                arg1_tp, arg2_tp, res_tp, nelems));
    });
    return comp_ev;
}

template <typename fnT, typename T1, typename T2> struct EqualContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename EqualOutputType<T1, T2>::value_type, void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = equal_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct EqualTypeMapFactory
{
    /*! @brief get typeid for output type of operator()==(x, y), always bool */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename EqualOutputType<T1, T2>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class equal_strided_strided_kernel;

typedef sycl::event (*equal_strided_impl_fn_ptr_t)(
    sycl::queue,
    size_t,
    int,
    const py::ssize_t *,
    const char *,
    py::ssize_t,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

template <typename argTy1, typename argTy2>
sycl::event
equal_strided_impl(sycl::queue exec_q,
                   size_t nelems,
                   int nd,
                   const py::ssize_t *shape_and_strides,
                   const char *arg1_p,
                   py::ssize_t arg1_offset,
                   const char *arg2_p,
                   py::ssize_t arg2_offset,
                   char *res_p,
                   py::ssize_t res_offset,
                   const std::vector<sycl::event> &depends,
                   const std::vector<sycl::event> &additional_depends)
{
    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename EqualOutputType<argTy1, argTy2>::value_type;

        using IndexerT =
            typename dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;

        IndexerT indexer{nd, arg1_offset, arg2_offset, res_offset,
                         shape_and_strides};

        const argTy1 *arg1_tp = reinterpret_cast<const argTy1 *>(arg1_p);
        const argTy2 *arg2_tp = reinterpret_cast<const argTy2 *>(arg2_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<
            equal_strided_strided_kernel<argTy1, argTy2, resTy, IndexerT>>(
            {nelems}, EqualStridedFunctor<argTy1, argTy2, resTy, IndexerT>(
                          arg1_tp, arg2_tp, res_tp, indexer));
    });
    return comp_ev;
}

template <typename fnT, typename T1, typename T2> struct EqualStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<
                          typename EqualOutputType<T1, T2>::value_type, void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = equal_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT1, typename argT2, typename resT>
class equal_matrix_row_broadcast_sg_krn;

typedef sycl::event (*equal_contig_matrix_contig_row_broadcast_impl_fn_ptr_t)(
    sycl::queue,
    std::vector<sycl::event> &,
    size_t,
    size_t,
    const char *,
    py::ssize_t,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename argT1, typename argT2, typename resT>
using EqualContigMatrixContigRowBroadcastingFunctor =
    elementwise_common::BinaryContigMatrixContigRowBroadcastingFunctor<
        argT1,
        argT2,
        resT,
        EqualFunctor<argT1, argT2, resT>>;

template <typename argT1, typename argT2, typename resT>
sycl::event equal_contig_matrix_contig_row_broadcast_impl(
    sycl::queue exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    py::ssize_t mat_offset,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    py::ssize_t vec_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = (mat[i,j] == vec[j])
    py::ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    const argT1 *mat = reinterpret_cast<const argT1 *>(mat_p) + mat_offset;
    const argT2 *vec = reinterpret_cast<const argT2 *>(vec_p) + vec_offset;
    resT *res = reinterpret_cast<resT *>(res_p) + res_offset;

    const auto &dev = exec_q.get_device();
    const auto &sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    // Get device-specific kernel info max_sub_group_size
    size_t max_sgSize =
        *(std::max_element(std::begin(sg_sizes), std::end(sg_sizes)));

    size_t n1_padded = n1 + max_sgSize;
    argT2 *padded_vec = sycl::malloc_device<argT2>(n1_padded, exec_q);

    if (padded_vec == nullptr) {
        throw std::runtime_error("Could not allocate memory on the device");
    }
    sycl::event make_padded_vec_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends); // ensure vec contains actual data
        cgh.parallel_for({n1_padded}, [=](sycl::id<1> id) {
            auto i = id[0];
            padded_vec[i] = vec[i % n1];
        });
    });

    // sub-group spans work-items [I, I + sgSize)
    // base = ndit.get_global_linear_id() - sg.get_local_id()[0]
    // Generically, sg.load( &mat[base]) may load arrays from
    // different rows of mat. The start corresponds to row (base / n0)
    // We read sg.load(&padded_vec[(base / n0)]). The vector is padded to
    // ensure that reads are accessible

    size_t lws = 64;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(make_padded_vec_ev);

        auto lwsRange = sycl::range<1>(lws);
        size_t n_elems = n0 * n1;
        size_t n_groups = (n_elems + lws - 1) / lws;
        auto gwsRange = sycl::range<1>(n_groups * lws);

        cgh.parallel_for<
            class equal_matrix_row_broadcast_sg_krn<argT1, argT2, resT>>(
            sycl::nd_range<1>(gwsRange, lwsRange),
            EqualContigMatrixContigRowBroadcastingFunctor<argT1, argT2, resT>(
                mat, padded_vec, res, n_elems, n1));
    });

    sycl::event tmp_cleanup_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_ev);
        sycl::context ctx = exec_q.get_context();
        cgh.host_task([ctx, padded_vec]() { sycl::free(padded_vec, ctx); });
    });
    host_tasks.push_back(tmp_cleanup_ev);

    return comp_ev;
}

template <typename fnT, typename T1, typename T2>
struct EqualContigMatrixContigRowBroadcastFactory
{
    fnT get()
    {
        using resT = typename EqualOutputType<T1, T2>::value_type;
        if constexpr (std::is_same_v<resT, void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    equal_contig_matrix_contig_row_broadcast_impl<T1, T2, resT>;
                return fn;
            }
        }
    }
};

typedef sycl::event (*equal_contig_row_contig_matrix_broadcast_impl_fn_ptr_t)(
    sycl::queue,
    std::vector<sycl::event> &,
    size_t,
    size_t,
    const char *,
    py::ssize_t,
    const char *,
    py::ssize_t,
    char *,
    py::ssize_t,
    const std::vector<sycl::event> &);

template <typename argT1, typename argT2, typename resT>
sycl::event equal_contig_row_contig_matrix_broadcast_impl(
    sycl::queue exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    py::ssize_t vec_offset,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    py::ssize_t mat_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = (mat[i,j] == vec[j])
    py::ssize_t res_offset,
    const std::vector<sycl::event> &depends = {})
{
    return equal_contig_matrix_contig_row_broadcast_impl<argT2, argT1, resT>(
        exec_q, host_tasks, n0, n1, mat_p, mat_offset, vec_p, vec_offset, res_p,
        res_offset, depends);
};

template <typename fnT, typename T1, typename T2>
struct EqualContigRowContigMatrixBroadcastFactory
{
    fnT get()
    {
        using resT = typename EqualOutputType<T1, T2>::value_type;
        if constexpr (std::is_same_v<resT, void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    equal_contig_row_contig_matrix_broadcast_impl<T1, T2, resT>;
                return fn;
            }
        }
    }
};

} // namespace equal
} // namespace kernels
} // namespace tensor
} // namespace dpctl
