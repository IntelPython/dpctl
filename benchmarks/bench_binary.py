import dpctl
import dpctl.tensor as dpt

import argparse


def get_sizes(n):
    s = []
    m = 8192
    while m < n:
        s.append(m)
        m *= 2
    s.append(n)
    return s


def run_bench(q, reps, n_max, dtype, op):
    run_bench2(q, reps, n_max, dtype, dtype, op)


def run_bench2(q, reps, n_max, dtype1, dtype2, op):
    x1 = dpt.ones(n_max, dtype=dtype1, sycl_queue=q)
    x2 = dpt.ones(n_max, dtype=dtype2, sycl_queue=q)
    r = op(x1, x2)

    max_bytes = (x1.nbytes + x2.nbytes + r.nbytes)

    times_res = []

    for n in get_sizes(n_max):
        x1_n = x1[:n]
        x2_n = x2[:n]
        r_n = r[:n]
        n_bytes = x1_n.nbytes + x2_n.nbytes + r_n.nbytes

        n_iters = int((max_bytes / n_bytes) * reps)

        while True:
            timer = dpctl.SyclTimer(device_timer="order_manager", time_scale=1e9)
            with timer(q):
                for _ in range(n_iters):
                    op(x1_n, x2_n, out=r_n)
                
            dev_dt = timer.dt.device_dt
            if dev_dt > 0:
                times_res.append((n, dev_dt / n_iters))
                break

    return times_res


def build_dtype_code(dt):
    return dt.kind + str(dt.itemsize)


def time_binary_supported_types(q, n_iters, n_values, op):
    dtypes = [list(map(dpt.dtype, sig.split('->')[0])) for sig in op.types]
    op_name = op.name_
    for dt1, dt2 in dtypes:
        r = run_bench2(q, n_iters, n_values, dt1, dt2, op)
        dtype_code1 = build_dtype_code(dt1)
        dtype_code2 = build_dtype_code(dt2)
        print(f"    '{op_name},{dtype_code1},{dtype_code2}': {r},")        


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Binary functions microbencharking tool"
    )
    driver_alg = parser.add_argument_group(title="Function selection arguments")
    driver_alg.add_argument(
        "-f", help="Functions to benchmark",
        dest="funcs",
        choices=[
            "all",
            "add", "subtract", "divide", "multiply",
            "floor_divide", "remainder",
            "hypot", "logaddexp", "pow", "atan2", "nextafter", "copysign",
            "less", "less_equal", "greater", "greater_equal", "equal", "not_equal",
            "minimum", "maximum",
            "bitwise_and", "bitwise_or", "bitwise_xor",
            "bitwise_left_shift", "bitwise_right_shift",
            "logical_and", "logical_or", "logical_xor"
        ],
        nargs="+",
        default="all"
    )
    driver_sizes = parser.add_argument_group(title="Sizes and repetition parameters")
    driver_sizes.add_argument(
        "-n", help="Input size",
        dest="n_values",
        default=2**27,
        type=int
    )
    driver_sizes.add_argument(
        "-r", help="Repetitions over which time is averaged",
        dest="n_iters",
        default=5,
        type=int
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    q = dpctl.SyclQueue(property='enable_profiling')
    n_iters = args.n_iters
    n_values = args.n_values
    if args.funcs == "all":
        f_list = [
            dpt.add, dpt.multiply, dpt.divide, dpt.subtract,
            dpt.floor_divide, dpt.remainder,
            dpt.hypot, dpt.logaddexp, dpt.pow, dpt.atan2, dpt.nextafter, dpt.copysign,
            dpt.less, dpt.less_equal, dpt.greater, dpt.greater_equal, dpt.equal, dpt.not_equal,
            dpt.minimum, dpt.maximum, dpt.bitwise_and, dpt.bitwise_or, dpt.bitwise_xor,
            dpt.bitwise_left_shift, dpt.bitwise_right_shift, dpt.logical_and, dpt.logical_or, dpt.logical_xor
        ]
    else:
        f_list = []
        for fn in args.funcs:

            if fn != "all" and hasattr(dpt, fn):
                f_list.append(getattr(dpt, fn))

    for f in f_list:
        time_binary_supported_types(q, n_iters, n_values, f)
