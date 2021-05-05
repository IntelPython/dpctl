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

template <typename T>
sycl::event zeros_contiguous(sycl::queue &exec_queue,
                             usm_array &ary,
                             const std::vector<sycl::event> &depends = {})
{
    const size_t *shape = ary.get_shape_ptr();
    const int nd = ary.ndim();

    CIndexer_vector ind_vec(nd);
    const std::ptrdiff_t n_elems = ind_vec.size(shape);

    T *p = reinterpret_cast<T *>(ary.get_data_ptr());

    sycl::event ev1 = exec_queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for(sycl::range<1>(n_elems),
                         [=](sycl::id<1> idx) { p[idx[0]] = T(0); });
    });

    return ev1;
}

template <typename T>
sycl::event zeros_generic(sycl::queue &exec_queue,
                          usm_array &ary,
                          const std::vector<sycl::event> &depends = {})
{
    T *p = reinterpret_cast<T *>(ary.get_data_ptr());
    const size_t *shape = ary.get_shape_ptr();
    const std::ptrdiff_t *strides = ary.get_strides_ptr();
    const int nd = ary.ndim();
    const int typenum = ary.typenum();
    const int flags = ary.flags();

    CIndexer_vector ind_vec(nd);
    const std::ptrdiff_t n_elems = ind_vec.size(shape);

    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
    sycl::buffer<size_t, 1> shape_buf(shape, sycl::range<1>(nd), props);
    sycl::buffer<std::ptrdiff_t, 1> strides_buf(strides, sycl::range<1>(nd),
                                                props);
    sycl::buffer<CIndexer_vector, 1> indexer_buf(&ind_vec, 1, props);

    sycl::event ev1 = exec_queue.submit([&](sycl::handler &cgh) {
        sycl::accessor shape_acc(shape_buf, cgh, sycl::read_only);
        sycl::accessor strides_acc(strides_buf, cgh, sycl::read_only);
        sycl::accessor indexer_acc(indexer_buf, cgh, sycl::read_only);

        cgh.depends_on(depends);
        cgh.parallel_for(sycl::range<1>(n_elems), [=](sycl::id<1> idx) {
            std::ptrdiff_t disp(0);
            indexer_acc[0].get_displacement(idx[0], shape_acc, strides_acc,
                                            disp);
            p[disp] = T(0);
        });
    });

    return ev1;
}

} // namespace details
} // namespace constructors
} // namespace usm_array
