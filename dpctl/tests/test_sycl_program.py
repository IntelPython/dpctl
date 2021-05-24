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

"""Defines unit test cases for the SyclProgram and SyclKernel classes
"""

import os
import unittest

import dpctl
import dpctl.program as dpctl_prog

from ._helper import has_gpu


@unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
class TestProgramFromOCLSource(unittest.TestCase):
    def test_create_program_from_source(self):
        oclSrc = "                                                             \
        kernel void add(global int* a, global int* b, global int* c) {         \
            size_t index = get_global_id(0);                                   \
            c[index] = a[index] + b[index];                                    \
        }                                                                      \
        kernel void axpy(global int* a, global int* b, global int* c, int d) { \
            size_t index = get_global_id(0);                                   \
            c[index] = a[index] + d*b[index];                                  \
        }"
        q = dpctl.SyclQueue("opencl:gpu")
        prog = dpctl_prog.create_program_from_source(q, oclSrc)
        self.assertIsNotNone(prog)

        self.assertTrue(prog.has_sycl_kernel("add"))
        self.assertTrue(prog.has_sycl_kernel("axpy"))

        addKernel = prog.get_sycl_kernel("add")
        axpyKernel = prog.get_sycl_kernel("axpy")

        self.assertEqual(addKernel.get_function_name(), "add")
        self.assertEqual(axpyKernel.get_function_name(), "axpy")
        self.assertEqual(addKernel.get_num_args(), 3)
        self.assertEqual(axpyKernel.get_num_args(), 4)


@unittest.skipUnless(has_gpu(), "No OpenCL GPU queues available")
class TestProgramFromSPRIV(unittest.TestCase):
    def test_create_program_from_spirv(self):

        CURR_DIR = os.path.dirname(os.path.abspath(__file__))
        spirv_file = os.path.join(CURR_DIR, "input_files/multi_kernel.spv")
        with open(spirv_file, "rb") as fin:
            spirv = fin.read()
            q = dpctl.SyclQueue("opencl:gpu")
            prog = dpctl_prog.create_program_from_spirv(q, spirv)
            self.assertIsNotNone(prog)
            self.assertTrue(prog.has_sycl_kernel("add"))
            self.assertTrue(prog.has_sycl_kernel("axpy"))

            addKernel = prog.get_sycl_kernel("add")
            axpyKernel = prog.get_sycl_kernel("axpy")

            self.assertEqual(addKernel.get_function_name(), "add")
            self.assertEqual(axpyKernel.get_function_name(), "axpy")
            self.assertEqual(addKernel.get_num_args(), 3)
            self.assertEqual(axpyKernel.get_num_args(), 4)


@unittest.skipUnless(
    has_gpu(backend=dpctl.backend_type.level_zero),
    "No Level0 GPU queues available",
)
class TestProgramForLevel0GPU(unittest.TestCase):

    import sys

    def test_create_program_from_spirv(self):
        CURR_DIR = os.path.dirname(os.path.abspath(__file__))
        spirv_file = os.path.join(CURR_DIR, "input_files/multi_kernel.spv")
        with open(spirv_file, "rb") as fin:
            spirv = fin.read()
            q = dpctl.SyclQueue("level_zero:gpu")
            dpctl_prog.create_program_from_spirv(q, spirv)

    @unittest.expectedFailure
    def test_create_program_from_source(self):
        oclSrc = "                                                             \
        kernel void add(global int* a, global int* b, global int* c) {         \
            size_t index = get_global_id(0);                                   \
            c[index] = a[index] + b[index];                                    \
        }                                                                      \
        kernel void axpy(global int* a, global int* b, global int* c, int d) { \
            size_t index = get_global_id(0);                                   \
            c[index] = a[index] + d*b[index];                                  \
        }"
        q = dpctl.SyclQueue("level_zero:gpu")
        dpctl_prog.create_program_from_source(q, oclSrc)


if __name__ == "__main__":
    unittest.main()
