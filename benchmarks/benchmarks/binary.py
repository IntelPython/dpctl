import dpctl
import dpctl.tensor as dpt

from . import benchmark_utils as bench_utils

SHARED_QUEUE = dpctl.SyclQueue(property="enable_profiling")


class Binary:
    """Benchmark class for binary operations on SYCL devices."""

    timeout = 300.0

    def setup(self):
        """Setup the benchmark environment."""
        self.q = SHARED_QUEUE
        self.iterations = 1
        self.n_values = 10**8

    def run_bench(self, q, reps, n_max, dtype1, dtype2, op):
        """Run the benchmark for a specific function and dtype combination."""

        def get_sizes(n):
            s = []
            m = 8192
            while m < n:
                s.append(m)
                m *= 2
            s.append(n)
            return s

        x1 = dpt.ones(n_max, dtype=dtype1, sycl_queue=q)
        x2 = dpt.ones(n_max, dtype=dtype2, sycl_queue=q)
        r = op(x1, x2)

        max_bytes = x1.nbytes + x2.nbytes + r.nbytes
        times_res = []

        for n in get_sizes(n_max):
            x1_n = x1[:n]
            x2_n = x2[:n]
            r_n = r[:n]
            n_bytes = x1_n.nbytes + x2_n.nbytes + r_n.nbytes

            n_iters = int((max_bytes / n_bytes) * reps)

            while True:
                timer = dpctl.SyclTimer(
                    device_timer="order_manager", time_scale=1e9
                )
                with timer(q):
                    for _ in range(n_iters):
                        op(x1_n, x2_n, out=r_n)

                dev_dt = timer.dt.device_dt
                if dev_dt > 0:
                    times_res.append((n, dev_dt / n_iters))
                    break

        return times_res


binary_instance = Binary()
binary_instance.q = SHARED_QUEUE
binary_instance.iterations = 1
binary_instance.n_values = 10**8

function_list = [
    dpt.add,
    dpt.multiply,
    dpt.divide,
    dpt.subtract,
    dpt.floor_divide,
    dpt.remainder,
    dpt.hypot,
    dpt.logaddexp,
    dpt.pow,
    dpt.atan2,
    dpt.nextafter,
    dpt.copysign,
    dpt.less,
    dpt.less_equal,
    dpt.greater,
    dpt.greater_equal,
    dpt.equal,
    dpt.not_equal,
    dpt.minimum,
    dpt.maximum,
    dpt.bitwise_and,
    dpt.bitwise_or,
    dpt.bitwise_xor,
    dpt.bitwise_left_shift,
    dpt.bitwise_right_shift,
    dpt.logical_and,
    dpt.logical_or,
    dpt.logical_xor,
]

# Generate dtype combinations for each function
dtypes = {}
for fn in function_list:
    dtypes[fn] = [list(map(dpt.dtype, sig.split("->")[0])) for sig in fn.types]


# Dynamically create benchmark methods at the module level
def generate_benchmark_functions():
    """Dynamically create benchmark functions for each
    function and dtype combination.
    """
    for fn in function_list:
        fn_name = fn.name_
        for dtype1, dtype2 in dtypes[fn]:
            # Create unique function names
            method_name = f"time_{fn_name}_{dtype1.name}_{dtype2.name}"

            def benchmark_method(self, fn=fn, dtype1=dtype1, dtype2=dtype2):
                bench_utils.skip_unsupported_dtype(self.q, dtype1)
                return self.run_bench(
                    self.q,
                    self.iterations,
                    self.n_values,
                    dtype1,
                    dtype2,
                    fn,
                )

            benchmark_method.__name__ = method_name
            # Attach the new method to the Binary class
            setattr(Binary, method_name, benchmark_method)


# Generate the benchmark functions
generate_benchmark_functions()
