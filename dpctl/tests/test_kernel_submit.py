import ctypes
import dpctl
import array
import numpy as np

def test_create_program_from_source ():
     q = dpctl.get_current_queue()
     q.get_sycl_device().dump_device_info()
     ctx = q.get_sycl_context()
     oclSrc = "                                                                \
     kernel void add(global int* a, global int* b, global int* c) {            \
         size_t index = get_global_id(0);                                      \
         c[index] = a[index] + b[index];                                       \
     }                                                                         \
     kernel void axpy(global int* a, global int* b, global int* c, int d) {    \
         size_t index = get_global_id(0);                                      \
         c[index] = a[index] + d*b[index];                                     \
     }"
     prog = dpctl.create_program_from_source(q,oclSrc)
     print(prog)
     print(prog.has_sycl_kernel('add'))
     print(prog.has_sycl_kernel('axpy'))
     addKernel = prog.get_sycl_kernel('add')
     print(addKernel)
     print(addKernel.get_function_name())
     print(addKernel.get_num_args())
     axpyKernel = prog.get_sycl_kernel('axpy')
     print(axpyKernel)
     print(axpyKernel.get_function_name())
     print(axpyKernel.get_num_args())
     
     a = np.arange(1024)
     b = np.arange(1024)
     c = np.zeros(1024)
     d = 2
     print(a)
     args = []
     
     print(type(a.data))
     args.append(a.data)
     args.append(b.data)
     args.append(c.data)
     args.append(ctypes.c_int(d))
     
     r = [ 1024, 1, 1 ]
     
     q.submit(axpyKernel, args, r, r)
     # self.assertIsNotNone(prog)
     
     # self.assertTrue()
     # self.assertTrue(prog.has_sycl_kernel("axpy"))
    
test_create_program_from_source()
