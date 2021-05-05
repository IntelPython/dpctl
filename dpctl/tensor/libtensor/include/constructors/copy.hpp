#pragma once

#include "usm_array.hpp"
#include "utils/strided_iters.hpp"
#include <CL/sycl.hpp>

namespace usm_array
{
namespace constructors
{
namespace details
{

template <class DstAryTy, class SrcAryTy>
sycl::event copy_generic(sycl::queue &exec_queue,
                         usm_array &dst_ary,
                         usm_array &src_ary,
                         sycl::vector_class<sycl::event> depends = {})
{
    DstAryTy *dst_p = reinterpret_cast<DstAryTy *>(dst_ary.get_data_ptr());
    const size_t *dst_shape = dst_ary.get_shape_ptr();
    const std::ptrdiff_t *dst_strides = dst_ary.get_strides_ptr();
    const int dst_nd = dst_ary.ndim();
    const int dst_typenum = dst_ary.typenum();
    const int dst_flags = dst_ary.flags();

    SrcAryTy *src_p = reinterpret_cast<SrcAryTy *>(src_ary.get_data_ptr());
    const size_t *src_shape = src_ary.get_shape_ptr();
    const std::ptrdiff_t *src_strides = src_ary.get_strides_ptr();
    const int src_nd = src_ary.ndim();
    const int src_typenum = src_ary.typenum();
    const int src_flags = src_ary.flags();

    /* nd must be the same, shapes must be the same
     */
    CIndexer_vector ind_vec(dst_nd);
    const std::ptrdiff_t n_elems = ind_vec.size(dst_shape);

    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
    sycl::buffer<size_t, 1> shape_buf(dst_shape, sycl::range<1>(dst_nd));
    sycl::buffer<std::ptrdiff_t, 1> dst_strides_buf(dst_strides,
                                                    sycl::range<1>(dst_nd));
    sycl::buffer<std::ptrdiff_t, 1> src_strides_buf(src_strides,
                                                    sycl::range<1>(src_nd));
    sycl::buffer<CIndexer_vector, 1> indexer_buf(&ind_vec, 1, props);

    sycl::event copy_event = exec_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor shape_acc(shape_buf, cgh, sycl::read_only);
        sycl::accessor src_strides_acc(src_strides_buf, cgh, sycl::read_only);
        sycl::accessor dst_strides_acc(dst_strides_buf, cgh, sycl::read_only);
        sycl::accessor indexer_acc(indexer_buf, cgh, sycl::read_only);
        cgh.depends_on(depends);

        cgh.parallel_for(sycl::range<1>(n_elems), [=](sycl::id<1> idx) {
            std::ptrdiff_t dst_disp(0), src_disp(0);
            indexer_acc[0].get_displacement(idx[0], shape_acc, dst_strides_acc,
                                            src_strides_acc, dst_disp,
                                            src_disp);
            dst_p[dst_disp] = static_cast<DstAryTy>(src_p[src_disp]);
        });
    });

    return copy_event;
}

template <class DstAryTy, class SrcAryTy>
sycl::event
copy_generic_to_c_contig(sycl::queue &exec_queue,
                         usm_array &dst_ary,
                         usm_array &src_ary,
                         sycl::vector_class<sycl::event> depends = {})
{
    DstAryTy *dst_p = reinterpret_cast<DstAryTy *>(dst_ary.get_data_ptr());
    const size_t *dst_shape = dst_ary.get_shape_ptr();
    const int dst_nd = dst_ary.ndim();
    const int dst_typenum = dst_ary.typenum();
    const int dst_flags = dst_ary.flags();

    SrcAryTy *src_p = reinterpret_cast<SrcAryTy *>(src_ary.get_data_ptr());
    const size_t *src_shape = src_ary.get_shape_ptr();
    const std::ptrdiff_t *src_strides = src_ary.get_strides_ptr();
    const int src_nd = src_ary.ndim();
    const int src_typenum = src_ary.typenum();
    const int src_flags = src_ary.flags();

    /* nd must be the same, shapes must be the same
     */
    CIndexer_vector ind_vec(dst_nd);
    const std::ptrdiff_t n_elems = ind_vec.size(dst_shape);

    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
    sycl::buffer<size_t, 1> shape_buf(dst_shape, sycl::range<1>(dst_nd));
    sycl::buffer<std::ptrdiff_t, 1> src_strides_buf(src_strides,
                                                    sycl::range<1>(src_nd));
    sycl::buffer<CIndexer_vector, 1> indexer_buf(&ind_vec, 1, props);

    sycl::event copy_event = exec_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor shape_acc(shape_buf, cgh, sycl::read_only);
        sycl::accessor src_strides_acc(src_strides_buf, cgh, sycl::read_only);
        sycl::accessor indexer_acc(indexer_buf, cgh, sycl::read_only);
        cgh.depends_on(depends);

        cgh.parallel_for(sycl::range<1>(n_elems), [=](sycl::id<1> idx) {
            std::ptrdiff_t dst_disp(idx.get(0)), src_disp(0);
            indexer_acc[0].get_displacement(idx[0], shape_acc, src_strides_acc,
                                            src_disp);
            dst_p[dst_disp] = static_cast<DstAryTy>(src_p[src_disp]);
        });
    });

    return copy_event;
}

template <class DstAryTy, class SrcAryTy>
sycl::event copy_generic_to_host(sycl::queue &exec_queue,
                                 strided_array &dst_host_ary,
                                 usm_array &src_ary,
                                 sycl::vector_class<sycl::event> depends = {})
{
    DstAryTy *dst_host_p =
        reinterpret_cast<DstAryTy *>(dst_host_ary.get_data_ptr());
    const size_t *dst_shape = dst_host_ary.get_shape_ptr();
    const int dst_nd = dst_host_ary.ndim();
    const std::ptrdiff_t *dst_strides = dst_host_ary.get_strides_ptr();
    const int dst_typenum = dst_host_ary.typenum();
    const int dst_flags = dst_host_ary.flags();

    SrcAryTy *src_usm_p = reinterpret_cast<SrcAryTy *>(src_ary.get_data_ptr());
    const size_t *src_shape = src_ary.get_shape_ptr();
    const std::ptrdiff_t *src_strides = src_ary.get_strides_ptr();
    const int src_nd = src_ary.ndim();
    const int src_typenum = src_ary.typenum();
    const int src_flags = src_ary.flags();

    /* nd must be the same, shapes must be the same
     */
    CIndexer_vector ind_vec(dst_nd);
    const std::ptrdiff_t n_elems = ind_vec.size(dst_shape);

    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
    sycl::buffer<size_t, 1> shape_buf(dst_shape, sycl::range<1>(dst_nd));
    sycl::buffer<std::ptrdiff_t, 1> dst_strides_buf(dst_strides,
                                                    sycl::range<1>(dst_nd));
    sycl::buffer<std::ptrdiff_t, 1> src_strides_buf(src_strides,
                                                    sycl::range<1>(src_nd));
    sycl::buffer<CIndexer_vector, 1> indexer_buf(&ind_vec, 1, props);

    /*
     * sycl::buffer needs a pointer to the start of the host block of memory,
     * which need not be dst_host_p, we need to compute smallest displacement
     * based on strides
     */
    std::ptrdiff_t min_disp = 0, max_disp = 0;
    for (int i = 0; i < dst_nd; ++i) {
        const auto sh_i = dst_shape[i] - 1;
        const auto str_i = dst_strides[i];
        const auto disp_delta = str_i * sh_i;
        if (str_i > 0) {
            max_disp += disp_delta;
        }
        else {
            min_disp += disp_delta;
        }
    }
    // It is important to not use use_host_ptr property here
    // we must release buffer
    sycl::buffer<DstAryTy, 1> dst_host_buf(dst_host_p + min_disp,
                                           sycl::range<1>(max_disp - min_disp));

    sycl::event copy_event = exec_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor shape_acc(shape_buf, cgh, sycl::read_only);
        sycl::accessor dst_strides_acc(dst_strides_buf, cgh, sycl::read_only);
        sycl::accessor src_strides_acc(src_strides_buf, cgh, sycl::read_only);
        sycl::accessor indexer_acc(indexer_buf, cgh, sycl::read_only);
        sycl::accessor dst_host_ary_acc(dst_host_buf, cgh);
        cgh.depends_on(depends);

        cgh.parallel_for(sycl::range<1>(n_elems), [=](sycl::id<1> idx) {
            std::ptrdiff_t dst_disp(0), src_disp(0);
            indexer_acc[0].get_displacement(idx[0], shape_acc, dst_strides_acc,
                                            src_strides_acc, dst_disp,
                                            src_disp);
            dst_host_ary_acc[dst_disp - min_disp] =
                static_cast<DstAryTy>(src_usm_p[src_disp]);
        });
    });
    // Must ensure the kernel finished otherwise
    // since we can not guarantee that destination host
    // pointer is going to be alive when the kernel execution
    // is completed
    copy_event.wait_and_throw();

    return copy_event;
}

} // namespace details
} // namespace constructors
} // namespace usm_array
