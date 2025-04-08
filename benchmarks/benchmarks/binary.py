import dpctl
import dpctl.tensor as dpt

class Binary:

    def setup(self):
      
        self.q = dpctl.SyclQueue(property='enable_profiling')
        self.n_iters = 1
        self.n_values = 2**27

        f_list = [
                dpt.add, dpt.multiply, dpt.divide, dpt.subtract,
                dpt.floor_divide, dpt.remainder,
                dpt.hypot, dpt.logaddexp, dpt.pow, dpt.atan2, dpt.nextafter, dpt.copysign,
                dpt.less, dpt.less_equal, dpt.greater, dpt.greater_equal, dpt.equal, dpt.not_equal,
                dpt.minimum, dpt.maximum, dpt.bitwise_and, dpt.bitwise_or, dpt.bitwise_xor,
                dpt.bitwise_left_shift, dpt.bitwise_right_shift, dpt.logical_and, dpt.logical_or, dpt.logical_xor
            ]

        for fn in f_list:
            method_name = 'time_' + fn.name_
            setattr(self, method_name, fn)


        for f in f_list:
            dtypes = [list(map(dpt.dtype, sig.split('->')[0])) for sig in op.types]
            for dt1, dt2 in dtypes:
                self.run_bench2(self.q, self.n_iters, self.n_values, dt1, dt2, f)


    def get_sizes(self, n):
        s = []
        m = 8192
        while m < n:
            s.append(m)
            m *= 2
        s.append(n)
        return s


    def run_bench(self, q, reps, n_max, dtype, op):
        self.run_bench2(q, reps, n_max, dtype, dtype, op)


    def run_bench2(self, q, reps, n_max, dtype1, dtype2, op):
        x1 = dpt.ones(n_max, dtype=dtype1, sycl_queue=q)
        x2 = dpt.ones(n_max, dtype=dtype2, sycl_queue=q)
        r = op(x1, x2)

        max_bytes = (x1.nbytes + x2.nbytes + r.nbytes)

        times_res = []

        for n in self.get_sizes(n_max):
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


    def build_dtype_code(self, dt):
        return dt.kind + str(dt.itemsize)
