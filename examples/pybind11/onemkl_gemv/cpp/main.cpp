#include "cg_solver.hpp"
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <oneapi/mkl.hpp>

using T = double;

int main(int argc, char *argv[])
{
    size_t n = 1000;
    size_t rank = 16;

    if (argc > 1) {
        n = std::stoi(argv[1]);
    }

    if (argc > 2) {
        rank = std::stoi(argv[2]);
    }

    std::cout << "Solving " << n << " by " << n << " diagonal system with rank-"
              << rank << " perturbation." << std::endl;

    sycl::queue q;

    // USM allocation for data needed by program
    size_t buf_size = n * n + rank * n + 2 * n;
    T *buf = sycl::malloc_device<T>(buf_size, q);
    sycl::event memset_ev = q.fill<T>(buf, T(0), buf_size);

    T *Amat = buf;
    T *umat = buf + n * n;
    T *bvec = umat + rank * n;
    T *sol_vec = bvec + n;

    sycl::event set_diag_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on({memset_ev});
        cgh.parallel_for({n}, [=](sycl::id<1> id) {
            auto i = id[0];
            Amat[i * (n + 1)] = T(1);
        });
    });

    oneapi::mkl::rng::philox4x32x10 engine(q, 7777);
    oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::icdf>
        distr(0.0, 1.0);

    // populate umat and bvec in one call
    sycl::event umat_rand_ev =
        oneapi::mkl::rng::generate(distr, engine, n * rank + n, umat);

    sycl::event syrk_ev = oneapi::mkl::blas::row_major::syrk(
        q, oneapi::mkl::uplo::U, oneapi::mkl::transpose::N, n, rank, T(1), umat,
        rank, T(1), Amat, n, {umat_rand_ev, set_diag_ev});

    // need to transpose
    sycl::event transpose_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(syrk_ev);
        cgh.parallel_for({n * n}, [=](sycl::id<1> id) {
            size_t i = id[0];
            size_t i0 = i / n;
            size_t i1 = i - i0 * n;
            if (i0 > i1) {
                Amat[i] = Amat[i1 * n + i0];
            }
        });
    });

    q.wait();

    constexpr int reps = 6;

    std::vector<double> time;
    std::vector<int> conv_iters;

    time.reserve(reps);
    conv_iters.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        int conv_iter_count = cg_solver::cg_solve(q, n, Amat, bvec, sol_vec);
        auto end = std::chrono::high_resolution_clock::now();

        time.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count() *
            1e-06);

        conv_iters.push_back(conv_iter_count);
    }

    std::cout << "Converged in : ";
    for (auto &el : conv_iters) {
        std::cout << el << " , ";
    }
    std::cout << std::endl;

    std::cout << "Wall-clock cg_solve execution times: ";
    for (auto &el : time) {
        std::cout << el << " , ";
    }
    std::cout << std::endl;

    T *Ax = sycl::malloc_device<T>(2 * n + 1, q);
    T *delta = Ax + n;

    sycl::event gemv_ev = oneapi::mkl::blas::row_major::gemv(
        q, oneapi::mkl::transpose::N, n, n, T(1), Amat, n, sol_vec, 1, T(0), Ax,
        1);

    sycl::event sub_ev = oneapi::mkl::vm::sub(q, n, Ax, bvec, delta, {gemv_ev},
                                              oneapi::mkl::vm::mode::ha);

    T *n2 = delta + n;
    sycl::event dot_ev = oneapi::mkl::blas::row_major::dot(
        q, n, delta, 1, delta, 1, n2, {sub_ev});

    T n2_host{};
    q.copy<T>(n2, &n2_host, 1, {dot_ev}).wait_and_throw();

    std::cout << "Redisual norm squared: " << n2_host << std::endl;

    q.wait_and_throw();
    sycl::free(Ax, q);
    sycl::free(buf, q);

    return 0;
}
