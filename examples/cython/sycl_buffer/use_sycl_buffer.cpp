#include <CL/sycl.hpp>
#include "use_sycl_buffer.h"
#include <oneapi/mkl.hpp>
#include "dpctl_sycl_types.h"

int
c_columnwise_total(DPCTLSyclQueueRef q_ref, size_t n, size_t m, double *mat, double *ct) {

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    sycl::buffer<double, 1> mat_buffer = sycl::buffer(mat, sycl::range<1>(n * m));
    sycl::buffer<double, 1> ct_buffer = sycl::buffer(ct, sycl::range<1>(m));

    double *ones = reinterpret_cast<double *>(malloc(n * sizeof(double)));
    {
	sycl::buffer<double, 1> ones_buffer = sycl::buffer(ones, sycl::range<1>(n));

	try {
	    auto ev = q.submit([&](sycl::handler &cgh) {
				   auto ones_acc = ones_buffer.get_access<sycl::access::mode::read_write>(cgh);
				   cgh.fill(ones_acc, double(1.0));
			       });
	    
	    ev.wait_and_throw();
	}
	catch (sycl::exception const& e) {
	    std::cout << "\t\tCaught synchronous SYCL exception during fill:\n"
		      << e.what() << std::endl << "OpenCL status: " << e.get_cl_code() << std::endl;
	    goto cleanup;
	}

	try {
	    oneapi::mkl::blas::row_major::gemv(
		q,
		oneapi::mkl::transpose::trans,
		n, m, double(1.0), mat_buffer, m,
		ones_buffer, 1,
		double(0.0), ct_buffer, 1);
	    q.wait();
	}
	catch (sycl::exception const &e) {
	    std::cout << "\t\tCaught synchronous SYCL exception during GEMV:\n"
		      << e.what() << std::endl << "OpenCL status: " << e.get_cl_code() << std::endl;
	    goto cleanup;
	}
    }
    
    free(ones);
    return 0;

  cleanup:
    free(ones);
    return -1;
}

inline size_t upper_multiple(size_t n, size_t wg) { return wg * ((n + wg - 1)/wg); }

int
c_columnwise_total_no_mkl(DPCTLSyclQueueRef q_ref, size_t n, size_t m, double *mat, double *ct) {

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    sycl::buffer<double, 2> mat_buffer = sycl::buffer(mat, sycl::range<2>(n, m));
    sycl::buffer<double, 1> ct_buffer = sycl::buffer(ct, sycl::range<1>(m));

    auto e = q.submit(
	[&](sycl::handler &h) {
	    sycl::accessor ct_acc {ct_buffer, h, sycl::write_only};
	    h.parallel_for(
		sycl::range<1>(m),
		[=](sycl::id<1> i){
		    ct_acc[i] = 0.0;
		});
	});

    constexpr size_t wg = 256;
    auto e2 = q.submit(
	[&](sycl::handler &h) {

	    sycl::accessor mat_acc {mat_buffer, h, sycl::read_only};
	    sycl::accessor ct_acc {ct_buffer, h};
	    h.depends_on(e);

	    sycl::range<2> global {upper_multiple(n, wg), m};
	    sycl::range<2> local {wg, 1};

	    h.parallel_for(
		sycl::nd_range<2>(global, local),
		[=](sycl::nd_item<2> it) {
		    size_t i = it.get_global_id(0);
		    size_t j = it.get_global_id(1);
		    double group_sum = sycl::ONEAPI::reduce(
			it.get_group(),
			(i < n) ? mat_acc[it.get_global_id()] : 0.0, 
			std::plus<double>()
			);
		    if (it.get_local_id(0) == 0) {
			sycl::ONEAPI::atomic_ref<
			    double,
			    sycl::ONEAPI::memory_order::relaxed,
			    sycl::ONEAPI::memory_scope::system,
			    sycl::access::address_space::global_space>(ct_acc[j]) += group_sum;
		    }
		});
	});

    e2.wait_and_throw();
    return 0;
}
