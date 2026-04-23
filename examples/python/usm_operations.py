#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2025 Intel Corporation
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

"""
Demonstrates SYCL USM memory usage in Python using dpctl.memory.
Includes allocation, host access, and host-device copying.
"""

import numpy as np

import dpctl.memory as dpmem


def usm_allocation():
    """
    Example demonstrating ways to allocate USM using dpctl.memory.
    """
    # allocate USM-shared byte-buffer
    ms = dpmem.MemoryUSMShared(16)

    # allocate USM-device byte-buffer
    md = dpmem.MemoryUSMDevice(16)

    # allocate USM-host byte-buffer
    mh = dpmem.MemoryUSMHost(16)

    # specify alignment
    mda = dpmem.MemoryUSMDevice(128, alignment=16)

    # allocate using given queue,
    # i.e. on the device and bound to the context stored in the queue
    mdq = dpmem.MemoryUSMDevice(256, queue=mda.sycl_queue)

    # information about device associate with USM buffer
    print("Allocation performed on device:")
    mda.sycl_queue.print_device_info()


def usm_host_access():
    """
    Example demonstrating that shared and host USM allocations are
    host-accessible and thus accessible from Python via buffer protocol.
    """
    # USM-shared and USM-host pointers are host-accessible,
    # meaning they are accessible from Python, therefore
    # they implement Pyton buffer protocol

    # allocate 1K of USM-shared buffer
    ms = dpmem.MemoryUSMShared(1024)

    # create memoryview into USM-shared buffer
    msv = memoryview(ms)

    # populate buffer from host one byte at a type
    for i in range(len(ms)):
        ir = i % 256
        msv[i] = ir**2 % 256

    mh = dpmem.MemoryUSMHost(64)
    mhv = memoryview(mh)

    # copy content of block of USM-shared buffer to
    # USM-host buffer
    mhv[:] = msv[78 : 78 + len(mh)]

    print("Byte-values of the USM-host buffer")
    print(list(mhv))

    # USM-device buffer is not host accessible
    md = dpmem.MemoryUSMDevice(16)
    try:
        mdv = memoryview(md)
    except Exception as e:
        print("")
        print(
            "An expected exception was raised during attempted construction of "
            "memoryview from USM-device memory object."
        )
        print(f"\t{e}")


def usm_host_device_copy():
    """
    Example demonstrating copying operations using dpctl.memory.
    """
    ms = dpmem.MemoryUSMShared(32)
    md = dpmem.MemoryUSMDevice(32)

    host_buf = np.random.randint(0, 42, dtype=np.uint8, size=32)

    # copy host byte-like object to USM-device buffer
    md.copy_from_host(host_buf)

    # copy USM-device buffer to USM-shared buffer in parallel using
    # sycl::queue::memcpy.
    ms.copy_from_device(md)

    # build numpy array reusing host-accessible USM-shared memory
    X = np.ndarray((len(ms),), buffer=ms, dtype=np.uint8)

    # Display Python object NumPy ndarray is viewing into
    print("numpy.ndarray.base: ", X.base)
    print("")

    # Print content of the view
    print("View..........: ", X)

    # Print content of the original host buffer
    print("host_buf......: ", host_buf)

    # use copy_to_host to retrieve memory of USM-device memory
    print("copy_to_host(): ", md.copy_to_host())


if __name__ == "__main__":
    import _runner as runner

    runner.run_examples("Memory examples for dpctl.", globals())
