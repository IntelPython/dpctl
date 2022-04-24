import numpy as np
import sycl_gemm

import dpctl
import dpctl.tensor as dpt


def empty_like(A):
    return dpt.empty(A.shape, A.dtype, device=A.device)


def chebyshev(A, b, x0, nIters, lMax, lMin, depends=[]):
    """Chebyshev iterative solver using SYCL routines"""
    d = (lMax + lMin) / 2
    c = (lMax - lMin) / 2

    x = dpt.copy(x0)
    exec_queue = A.sycl_queue
    assert exec_queue == x.sycl_queue
    Ax = empty_like(A[:, 0])
    r = empty_like(Ax)
    p = empty_like(Ax)

    e_x = dpctl.SyclEvent()
    he_dot, e_dot = sycl_gemm.gemv(
        exec_queue, A, x, Ax, depends=depends
    )  # Ax = A @ x
    he_sub, e_sub = sycl_gemm.sub(
        exec_queue, b, Ax, r, depends=[e_dot]
    )  # r = b - Ax
    r_ev = e_sub
    for i in range(nIters):
        z = r
        z_ev = r_ev
        if i == 0:
            p[:] = z
            alpha = 1 / d
            he_axbpy, e_axbpy = dpctl.SyclEvent(), dpctl.SyclEvent()
        elif i == 1:
            beta = 0.5 * (c * alpha) ** 2
            alpha = 1 / (d - beta / alpha)
            he_axbpy, e_axbpy = sycl_gemm.axbpy_inplace(
                exec_queue, 1, z, beta, p, depends=[z_ev]
            )  # p = z + beta * p
        else:
            beta = (c / 2 * alpha) ** 2
            alpha = 1 / (d - beta / alpha)
            he_axbpy, e_axbpy = sycl_gemm.axbpy_inplace(
                exec_queue, 1, z, beta, p, depends=[z_ev]
            )  # p = z + beta * p
        h_x, e_x = sycl_gemm.axbpy_inplace(
            exec_queue, alpha, p, 1, x, depends=[e_axbpy, e_x]
        )  # x = x + alpha * p
        he_dot, e_dot = sycl_gemm.gemv(
            exec_queue, A, x, Ax, depends=[e_x]
        )  # Ax = A @ x
        he_sub, e_sub = sycl_gemm.sub(
            exec_queue, b, Ax, r, depends=[e_dot]
        )  # r = b - Ax
        residual = sycl_gemm.norm_squared_blocking(
            exec_queue, r, depends=[e_sub]
        )  # residual = dot(r, r)
        if residual <= 1e-29:
            print(f"chebyshev: converged in {i} iters")
            break
    return x


def check_with_numpy(A, b):
    """Direct solver using numpy"""
    import numpy as np

    Anp = dpt.asnumpy(A)
    bnp = dpt.asnumpy(b)
    return np.linalg.solve(Anp, bnp)


def chebyshev_numpy(A, b, x0, nIters, lMax, lMin):
    """Chebyshev iterative solver using numpy"""
    d = (lMax + lMin) / 2
    c = (lMax - lMin) / 2

    x = x0

    Ax = A @ x
    r = b - Ax
    for i in range(nIters):
        z = r
        if i == 0:
            p = z
            alpha = 1 / d
        elif i == 1:
            beta = 0.5 * (c * alpha) ** 2
            alpha = 1 / (d - beta / alpha)
            p = z + beta * p
        else:
            beta = (c / 2 * alpha) ** 2
            alpha = 1 / (d - beta / alpha)
            p = z + beta * p
        x += alpha * p
        Ax = A @ x
        r = b - Ax
        residual = np.dot(r, r)
        if residual <= 1e-29:
            print(f"chebyshev_numpy: converged in {i} iters")
            break
    return x


if __name__ == "__main__":
    A = dpt.asarray(
        [[2, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]],
        dtype="d",
        usm_type="device",
    )
    dev = A.device
    b = dpt.asarray(
        [0.5, 0.7, 0.3, 0.1], dtype="d", usm_type="device", device=dev
    )
    x0 = b
    x = chebyshev(A, b, x0, 100, 4.0, 0.25)
    print(dpt.asnumpy(x))
    print(check_with_numpy(A, b))
    print(
        chebyshev_numpy(
            dpt.asnumpy(A), dpt.asnumpy(b), dpt.asnumpy(b), 100, 4.0, 0.25
        )
    )
