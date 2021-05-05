#include "constructors/zeros.hpp"
#include "usm_array.hpp"
#include "utils/strided_iters.hpp"
#include <CL/sycl.hpp>

int main_sycl_kernels(void)
{
    sycl::queue q{sycl::default_selector{}};
    DPCTLSyclQueueRef qref = reinterpret_cast<DPCTLSyclQueueRef>(&q);

    size_t shape[2] = {5, 5};
    CIndexer_vector ind(2);
    size_t nelems = ind.size(shape);

    int *data = sycl::malloc_device<int>(nelems, q);

    sycl::event pop_event =
        q.parallel_for(sycl::range<1>(nelems),
                       [=](sycl::id<1> idx) { data[idx[0]] = int(1); });

    size_t subshape[2] = {3, 3};
    std::ptrdiff_t strides[2] = {5, 1};
    usm_array::usm_array ary(reinterpret_cast<char *>(data + 6), 2, subshape,
                             strides, 0, 0, qref);

    sycl::event zeros_event =
        usm_array::constructors::details::zeros_generic<int>(q, ary,
                                                             {pop_event});

    size_t subshape2[2] = {3, 5};
    usm_array::usm_array ary2(reinterpret_cast<char *>(data + 5), 2, subshape2,
                              nullptr, 0, 0, qref);
    sycl::event zeros_event2 =
        usm_array::constructors::details::zeros_contiguous<int>(q, ary2,
                                                                {zeros_event});
    zeros_event2.wait();

    int *host_data = reinterpret_cast<int *>(malloc(nelems * sizeof(int)));
    q.memcpy(reinterpret_cast<void *>(host_data),
             reinterpret_cast<void *>(data), nelems * sizeof(int));
    q.wait();

    for (int i0 = 0; i0 < shape[0]; i0++) {
        for (int i1 = 0; i1 < shape[1]; i1++) {
            std::cout << host_data[i0 * shape[1] + i1] << " ";
        }
        std::cout << std::endl;
    }

    free(host_data);
    sycl::free(data, q);

    size_t shape1[5] = {2, 1, 5, 1, 3};
    std::ptrdiff_t strides1[5] = {15, 15, 3, 3, 1};
    std::ptrdiff_t disp(0);

    auto nd = simplify_iteration_stride(5, shape1, strides1, disp);

    std::cout << "Displacement " << disp << std::endl;
    std::cout << "Simplified nd = " << nd << std::endl;
    std::cout << "Simplified shape: ";
    for (int i = 0; i < nd; ++i) {
        std::cout << shape1[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Simplified strides: ";
    for (int i = 0; i < nd; ++i) {
        std::cout << strides1[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
