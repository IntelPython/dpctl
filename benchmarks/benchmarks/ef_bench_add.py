import dpctl
import dpctl.tensor as dpt
import dpctl.utils as dpu
import dpctl.tensor._tensor_elementwise_impl as tei

class Suite:

    def setup(self):
        self.q = dpctl.SyclQueue(property='enable_profiling')

        self.n = 2**26
        self.reps = 50

        self.dt = dpt.int8
        self.x1 = dpt.ones(self.n, dtype=self.dt, sycl_queue=q)
        self.x2 = dpt.ones(self.n, dtype=self.dt, sycl_queue=q)

        self.op1, self.op2 = dpt.add, tei._add

        self.r = self.op1(self.x1, self.x2)

        self.timer = dpctl.SyclTimer(device_timer="order_manager", time_scale=1e9)
        self.m = dpu.SequentialOrderManager[self.q]

    def time_ef_bench_add(self):
        with self.timer(self.q):
            for _ in range(self.reps):
                deps = self.m.submitted_events
                ht_e, c_e = self.op2(src1=self.x1,
                                src2=self.x2,
                                dst=self.r,
                                sycl_queue=self.q,
                                depends=deps)
                self.m.add_event_pair(ht_e, c_e)
        
        # ddt = self.timer.dt.device_dt
        # print((self.n, self.reps, self.dt))
        # print(ddt / self.reps)
        # print(dpt.max(dpt.abs(self.r - 2)))
