##===------------- test_sycl_program.py - dpctl  -------*- Python -*-------===##
##
##                      Data Parallel Control (dpctl)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
##
## \file
## Defines unit test cases for the SyclProgram and SyclKernel classes defined
##  in sycl_core.pyx.
##===----------------------------------------------------------------------===##

import dpctl
import unittest


@unittest.skipIf(not dpctl.has_sycl_platforms(), "No SYCL platforms available")
class TestProgramFromOCSource (unittest.TestCase):

    def test_create_program_from_source (self):
        q = dpctl.get_current_queue()
        oclSrc = "                                                             \
        kernel void add(global int* a, global int* b, global int* c) {         \
            size_t index = get_global_id(0);                                   \
            c[index] = a[index] + b[index];                                    \
        }                                                                      \
        kernel void axpy(global int* a, global int* b, global int* c, int d) { \
            size_t index = get_global_id(0);                                   \
            c[index] = a[index] + d*b[index];                                  \
        }"
        prog = dpctl.create_program_from_source(q,oclSrc)
        self.assertIsNotNone(prog)

        self.assertTrue(prog.has_sycl_kernel("add"))
        self.assertTrue(prog.has_sycl_kernel("axpy"))

        addKernel = prog.get_sycl_kernel('add')
        axpyKernel = prog.get_sycl_kernel('axpy')

        self.assertEqual(addKernel.get_function_name(),"add")
        self.assertEqual(axpyKernel.get_function_name(),"axpy")
        self.assertEqual(addKernel.get_num_args(), 3)
        self.assertEqual(axpyKernel.get_num_args(), 4)


if __name__ == '__main__':
    unittest.main()
