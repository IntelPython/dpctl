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
namespace add
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2>
struct AddContigFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    const size_t nelems_;

public:
    AddContigFunctor(const argT1 *inp1,
                     const argT2 *inp2,
                     resT *res,
                     const size_t n_elems)
        : in1(inp1), in2(inp2), out(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        /* Each work-item processes vec_sz elements, contiguous in memory */

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<argT1>::value || is_complex<argT2>::value) {
            std::uint8_t sgSize = ndit.get_sub_group().get_local_range()[0];
            size_t base = ndit.get_global_linear_id();

            base = (base / sgSize) * sgSize * n_vecs * vec_sz + (base % sgSize);
            for (size_t offset = base;
                 offset < std::min(nelems_, base + sgSize * (n_vecs * vec_sz));
                 offset += sgSize)
            {
                out[offset] = in1[offset] + in2[offset];
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
                    if constexpr (std::is_same_v<argT1, resT> &&
                                  std::is_same_v<argT2, resT>) {
                        res_vec = arg1_vec + arg2_vec;
                    }
                    else {
                        using dpctl::tensor::type_utils::vec_cast;

                        auto tmp = arg1_vec + arg2_vec;
                        res_vec = std::move(
                            vec_cast<resT, typename decltype(tmp)::element_type,
                                     vec_sz>(tmp));
                    }
                    sg.store<vec_sz>(out_ptrT(&out[base + it * sgSize]),
                                     res_vec);
                }
            }
            else {
                for (size_t k = base + sg.get_local_id()[0]; k < nelems_;
                     k += sgSize) {
                    out[k] = in1[k] + in2[k];
                }
            }
        }
    }
};

template <typename T1, typename T2> struct AddOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        td_ns::BinaryTypeMapEntry<T1, bool, T2, bool, bool>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::uint8_t,
                                  T2,
                                  std::uint8_t,
                                  std::uint8_t>,
        td_ns::
            BinaryTypeMapEntry<T1, std::int8_t, T2, std::int8_t, std::int8_t>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::uint16_t,
                                  T2,
                                  std::uint16_t,
                                  std::uint16_t>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::int16_t,
                                  T2,
                                  std::int16_t,
                                  std::int16_t>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::uint32_t,
                                  T2,
                                  std::uint32_t,
                                  std::uint32_t>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::int32_t,
                                  T2,
                                  std::int32_t,
                                  std::int32_t>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::uint64_t,
                                  T2,
                                  std::uint64_t,
                                  std::uint64_t>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::int64_t,
                                  T2,
                                  std::int64_t,
                                  std::int64_t>,
        td_ns::BinaryTypeMapEntry<T1, sycl::half, T2, sycl::half, sycl::half>,
        td_ns::BinaryTypeMapEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapEntry<T1, double, T2, double, double>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::complex<float>,
                                  T2,
                                  std::complex<float>,
                                  std::complex<float>>,
        td_ns::BinaryTypeMapEntry<T1,
                                  std::complex<double>,
                                  T2,
                                  std::complex<double>,
                                  std::complex<double>>,
        td_ns::DefaultEntry<void>>::result_type;
};

template <typename argT1,
          typename argT2,
          typename resT,
          unsigned int vec_sz,
          unsigned int n_vecs>
class add_contig_kernel;

typedef sycl::event (*add_contig_impl_fn_ptr_t)(
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
sycl::event add_contig_impl(sycl::queue exec_q,
                            size_t nelems,
                            const char *arg1_p,
                            py::ssize_t arg1_offset,
                            const char *arg2_p,
                            py::ssize_t arg2_offset,
                            char *res_p,
                            py::ssize_t res_offset,
                            const std::vector<sycl::event> &depends = {})
{
    sycl::event add_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        size_t lws = 64;
        constexpr unsigned int vec_sz = 4;
        constexpr unsigned int n_vecs = 2;
        const size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        using resTy = typename AddOutputType<argTy1, argTy2>::value_type;

        const argTy1 *arg1_tp =
            reinterpret_cast<const argTy1 *>(arg1_p) + arg1_offset;
        const argTy2 *arg2_tp =
            reinterpret_cast<const argTy2 *>(arg2_p) + arg2_offset;
        resTy *res_tp = reinterpret_cast<resTy *>(res_p) + res_offset;

        cgh.parallel_for<
            add_contig_kernel<argTy1, argTy2, resTy, vec_sz, n_vecs>>(
            sycl::nd_range<1>(gws_range, lws_range),
            AddContigFunctor<argTy1, argTy2, resTy, vec_sz, n_vecs>(
                arg1_tp, arg2_tp, res_tp, nelems));
    });
    return add_ev;
}

template <typename fnT, typename T1, typename T2> struct AddContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AddOutputType<T1, T2>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = add_contig_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2> struct AddTypeMapFactory
{
    /*! @brief get typeid for output type of std::add(T1 x, T2 y) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename AddOutputType<T1, T2>::value_type;
        ;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argT1,
          typename argT2,
          typename resT,
          typename ThreeOffsets_IndexerT>
struct AddStridedFunctor
{
private:
    const argT1 *in1 = nullptr;
    const argT2 *in2 = nullptr;
    resT *out = nullptr;
    ThreeOffsets_IndexerT three_offsets_indexer_;

public:
    AddStridedFunctor(const argT1 *inp1_tp,
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

        out[out_offset] = in1[inp1_offset] + in2[inp2_offset];
    }
};

template <typename T1, typename T2, typename resT, typename IndexerT>
class add_strided_strided_kernel;

typedef sycl::event (*add_strided_impl_fn_ptr_t)(
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
sycl::event add_strided_impl(sycl::queue exec_q,
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
    sycl::event abs_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        using resTy = typename AddOutputType<argTy1, argTy2>::value_type;

        using IndexerT =
            typename dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;

        IndexerT indexer{nd, arg1_offset, arg2_offset, res_offset,
                         shape_and_strides};

        const argTy1 *arg1_tp = reinterpret_cast<const argTy1 *>(arg1_p);
        const argTy2 *arg2_tp = reinterpret_cast<const argTy2 *>(arg2_p);
        resTy *res_tp = reinterpret_cast<resTy *>(res_p);

        cgh.parallel_for<
            add_strided_strided_kernel<argTy1, argTy2, resTy, IndexerT>>(
            {nelems}, AddStridedFunctor<argTy1, argTy2, resTy, IndexerT>(
                          arg1_tp, arg2_tp, res_tp, indexer));
    });
    return abs_ev;
}

template <typename fnT, typename T1, typename T2> struct AddStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AddOutputType<T1, T2>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = add_strided_impl<T1, T2>;
            return fn;
        }
    }
};

template <typename argT1, typename argT2, typename resT>
class add_matrix_vector_broadcast_sg_krn;

typedef sycl::event (*add_contig_matrix_contig_row_broadcast_impl_fn_ptr_t)(
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
sycl::event add_contig_matrix_contig_row_broadcast_impl(
    sycl::queue exec_q,
    std::vector<sycl::event> &host_tasks,
    size_t n0,
    size_t n1,
    const char *mat_p, // typeless pointer to (n0, n1) C-contiguous matrix
    py::ssize_t mat_offset,
    const char *vec_p, // typeless pointer to (n1,) contiguous row
    py::ssize_t vec_offset,
    char *res_p, // typeless pointer to (n0, n1) result C-contig. matrix,
                 //    res[i,j] = mat[i,j] + vec[j]
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
        size_t n_groups = (n0 * n1 + lws - 1) / lws;
        auto gwsRange = sycl::range<1>(n_groups * lws);

            cgh.parallel_for<class add_matrix_vector_broadcast_sg_krn<argT1, argT2, resT>>(
                sycl::nd_range<1>(gwsRange, lwsRange),
                [=](sycl::nd_item<1> ndit)
            {
                auto sg = ndit.get_sub_group();
                size_t gid = ndit.get_global_linear_id();

                size_t base = gid - sg.get_local_id()[0];

                using in_ptrT1 =
                    sycl::multi_ptr<const argT1,
                                    sycl::access::address_space::global_space>;
                using in_ptrT2 =
                    sycl::multi_ptr<const argT2,
                                    sycl::access::address_space::global_space>;
                using res_ptrT =
                    sycl::multi_ptr<resT,
                                    sycl::access::address_space::global_space>;

                const argT1 mat_el = sg.load(in_ptrT1(&mat[base]));
                const argT2 vec_el = sg.load(in_ptrT2(&padded_vec[base % n1]));

                resT res_el = mat_el + vec_el;

                sg.store(res_ptrT(&res[base]), res_el);
                }
            );
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
struct AddContigMatrixContigRowBroadcastFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename AddOutputType<T1, T2>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using resT = typename AddOutputType<T1, T2>::value_type;
            if constexpr (dpctl::tensor::type_utils::is_complex<T1>::value ||
                          dpctl::tensor::type_utils::is_complex<T2>::value ||
                          dpctl::tensor::type_utils::is_complex<resT>::value)
            {
                fnT fn = nullptr;
                return fn;
            }
            else {
                fnT fn =
                    add_contig_matrix_contig_row_broadcast_impl<T1, T2, resT>;
                return fn;
            }
        }
    }
};

} // namespace add
} // namespace kernels
} // namespace tensor
} // namespace dpctl
