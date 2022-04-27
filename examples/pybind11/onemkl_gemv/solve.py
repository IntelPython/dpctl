#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            he_axpby, e_axpby = dpctl.SyclEvent(), dpctl.SyclEvent()
        elif i == 1:
            beta = 0.5 * (c * alpha) ** 2
            alpha = 1 / (d - beta / alpha)
            he_axpby, e_axpby = sycl_gemm.axpby_inplace(
                exec_queue, 1, z, beta, p, depends=[z_ev]
            )  # p = z + beta * p
        else:
            beta = (c / 2 * alpha) ** 2
            alpha = 1 / (d - beta / alpha)
            he_axpby, e_axpby = sycl_gemm.axpby_inplace(
                exec_queue, 1, z, beta, p, depends=[z_ev]
            )  # p = z + beta * p
        h_x, e_x = sycl_gemm.axpby_inplace(
            exec_queue, alpha, p, 1, x, depends=[e_axpby, e_x]
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
    exec_queue.wait()  # wait for all host tasks to complete
    return x


def check_with_numpy(A, b):
    """Direct solver using numpy"""
    import numpy as np

    return np.linalg.solve(Anp, bnp)


def chebyshev_numpy(A, b, x0, nIters, lMax, lMin):
    """Chebyshev iterative solver using numpy"""
    d = (lMax + lMin) / 2
    c = (lMax - lMin) / 2

    x = x0

    Ax = np.dot(A, x)
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
        x = x + alpha * p
        Ax = np.dot(A, x)
        r = b - Ax
        residual = np.dot(r, r)
        if residual <= 1e-29:
            print(f"chebyshev_numpy: converged in {i} iters")
            break
    return x


def cg_solve(A, b):
    """
    Conjugate gradient solver for A @ x == b.

    Returns tuple: (x, converged)

    converged is False if solver has not converged, or the iteration number
    """
    exec_queue = A.sycl_queue
    x = dpt.zeros(b.shape, dtype=b.dtype)
    Ap = empty_like(x)

    all_host_tasks = []
    r = dpt.copy(b)
    p = dpt.copy(b)
    rsold = sycl_gemm.norm_squared_blocking(exec_queue, r)
    if rsold < 1e-20:
        return (b, 0)
    converged = False
    max_iters = b.shape[0]

    e_p = dpctl.SyclEvent()
    e_x = dpctl.SyclEvent()
    for i in range(max_iters):
        # Ap = A @ p
        he_dot, e_dot = sycl_gemm.gemv(exec_queue, A, p, Ap, depends=[e_p])
        all_host_tasks.append(he_dot)
        # alpha = rsold / dot(p, Ap)
        alpha = rsold / sycl_gemm.dot_blocking(
            exec_queue, p, Ap, depends=[e_dot]
        )
        # x = x + alpha * p
        he1_x_update, e1_x_update = sycl_gemm.axpby_inplace(
            exec_queue, alpha, p, 1, x, depends=[e_p, e_x]
        )
        all_host_tasks.append(he1_x_update)
        e_x = e1_x_update

        # r = r - alpha * Ap
        he2_r_update, e2_r_update = sycl_gemm.axpby_inplace(
            exec_queue, -alpha, Ap, 1, r, depends=[e_p]
        )
        all_host_tasks.append(he2_r_update)

        # rsnew = dot(r, r)
        rsnew = sycl_gemm.norm_squared_blocking(
            exec_queue, r, depends=[e2_r_update]
        )
        if rsnew < 1e-20:
            e1_x_update.wait()
            converged = i
            break
        beta = rsnew / rsold

        # p = r + beta * p
        he3_p_update, e3_p_update = sycl_gemm.axpby_inplace(
            exec_queue, 1, r, beta, p, depends=[e2_r_update]
        )

        rsold = rsnew
        all_host_tasks.append(he3_p_update)
        e_p = e3_p_update
        e_x = e1_x_update

    dpctl.SyclEvent.wait_for(all_host_tasks)
    return x, converged


def cg_solve_numpy(A, b):
    x = np.zeros_like(b)
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(r, r)
    converged = False
    max_iters = b.shape[0]

    for i in range(max_iters):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)

        if rsnew < 1e-20:
            converged = i
            break

        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return (x, converged)


if __name__ == "__main__":
    n = 32
    Anp = (
        2 * np.eye(n, n, k=0, dtype="d")
        + np.eye(n, n, k=1, dtype="d")
        + np.eye(n, n, k=-1, dtype="d")
    )
    bnp = np.geomspace(0.5, 2, n, dtype="d")
    # bounds on eigenvalues of cartan matrix are, needed only
    # for the Chebyshev solver
    lambda_max = 4
    lambda_min = 4 * np.square(np.sin(np.pi / (2 * (n + 2))))

    q = dpctl.SyclQueue(property="enable_profiling")
    q.print_device_info()
    A = dpt.asarray(Anp, dtype="d", usm_type="device", sycl_queue=q)
    dev = A.device
    b = dpt.asarray(bnp, dtype="d", usm_type="device", device=dev)
    x0 = b
    t = dpctl.SyclTimer()
    with t(dev.sycl_queue):
        x, conv = cg_solve(A, b)
    print("SYCL solver, 1st run: ", (conv, t.dt))
    with t(dev.sycl_queue):
        x, conv = cg_solve(A, b)
    print("SYCL solver, 2nd run: ", (conv, t.dt))

    x_ref = check_with_numpy(Anp, bnp)  # solve usign LU solver

    with t(dev.sycl_queue):
        x_np, conv = cg_solve_numpy(dpt.asnumpy(A), dpt.asnumpy(b))
    print("NumPy's powered CG solver: ", (conv, t.dt))
    print(
        "SYCL cg-solver solution close to reference: ",
        np.allclose(dpt.asnumpy(x), x_ref),
    )
    print(
        "NumPy's cg-solver solution close to reference: ",
        np.allclose(x_np, x_ref),
    )
