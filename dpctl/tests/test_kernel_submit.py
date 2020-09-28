import ctypes
import dpctl
import dpctl._memory as dpctl_mem
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

     abuf = dpctl_mem.MemoryUSMShared(1024*np.dtype('i').itemsize)
     bbuf = dpctl_mem.MemoryUSMShared(1024*np.dtype('i').itemsize)
     cbuf = dpctl_mem.MemoryUSMShared(1024*np.dtype('i').itemsize)
     a = np.ndarray((1024), buffer=abuf, dtype='i')
     b = np.ndarray((1024), buffer=bbuf, dtype='i')
     c = np.ndarray((1024), buffer=cbuf, dtype='i')
     a[:] = np.arange(1024)
     b[:] = np.arange(1024, 0, -1)
     c[:] = 0
     d = 2
     args = []
     
     args.append(a.base)
     args.append(b.base)
     args.append(c.base)
     args.append(ctypes.c_int(d))
     
     r = [ 1024, 1, 1 ]
     
     e = q.submit(axpyKernel, args, r, r)
     e.wait()

     print(c)
     print(a + d * b)
     # self.assertIsNotNone(prog)
     
     # self.assertTrue()
     # self.assertTrue(prog.has_sycl_kernel("axpy"))
    
test_create_program_from_source()
