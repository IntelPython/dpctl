import ctypes
import dpctl
import unittest
import dpctl._memory as dpctl_mem
import numpy as np

@unittest.skipIf(not dpctl.has_sycl_platforms(), "No SYCL platforms available")
class Test1DKernelSubmit (unittest.TestCase):

    def test_create_program_from_source (self):
        q = dpctl.get_current_queue()
        ctx = q.get_sycl_context()
        oclSrc = "                                                             \
        kernel void axpy(global int* a, global int* b, global int* c, int d) { \
            size_t index = get_global_id(0);                                   \
            c[index] = d*a[index] + b[index];                                  \
        }"
        prog = dpctl.create_program_from_source(q,oclSrc)
        axpyKernel = prog.get_sycl_kernel('axpy')

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

        r = [1024]

        q.submit(axpyKernel, args, r)
        self.assertTrue(np.allclose(c, a*d + b))

if __name__ == '__main__':
    unittest.main()
