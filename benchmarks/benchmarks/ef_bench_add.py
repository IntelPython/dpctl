import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as tei
import dpctl.utils as dpu


class EfBenchAdd:

    def time_ef_bench_add(self):
        q = dpctl.SyclQueue(property="enable_profiling")
        n = 2**26
        reps = 50

        dt = dpt.int8
        x1 = dpt.ones(n, dtype=dt, sycl_queue=q)
        x2 = dpt.ones(n, dtype=dt, sycl_queue=q)

        op1, op2 = dpt.add, tei._add

        r = op1(x1, x2)

        timer = dpctl.SyclTimer(device_timer="order_manager", time_scale=1e9)

        m = dpu.SequentialOrderManager[q]
        with timer(q):
            for _ in range(reps):
                deps = m.submitted_events
                ht_e, c_e = op2(
                    src1=x1, src2=x2, dst=r, sycl_queue=q, depends=deps
                )
                m.add_event_pair(ht_e, c_e)
