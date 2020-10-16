#include <CL/sycl.hpp>
#include "use_sycl_buffer.h"
#include "mkl_blas_sycl.hpp"
#include "mkl.h"
#include "dppl_sycl_types.h"

int
c_columnwise_total(DPPLSyclQueueRef q_ref, size_t n, size_t m, double *mat, double *ct) {

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
