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

import dpctl
import dpctl.memory


def is_root_device(d):
    """
    Returns True if d is an instance of SyclDevice that does not
    have a parent_device, or False otherwise.
    """
    if not isinstance(d, dpctl.SyclDevice):
        return False
    # d.parent_device is None for a root device,
    # or a SyclDevice object representing the parent device
    # for a sub-device
    return d.parent_device is None


def subdivide_root_cpu_device():
    """
    Create root CPU device, and equally partition it
    into smaller CPU devices 4 execution units each,
    and then further parition those subdevice into
    smaller sub-devices
    """
    cpu_d = dpctl.SyclDevice("cpu")
    print(
        "cpu_d is "
        + ("a root device." if is_root_device(cpu_d) else "not a root device.")
    )
    sub_devs = cpu_d.create_sub_devices(partition=4)
    print("Sub-device #EU: ", [d.max_compute_units for d in sub_devs])
    print("Sub-device is_root: ", [is_root_device(d) for d in sub_devs])
    print(
        "Sub-device parent is what we expected: ",
        [d.parent_device == cpu_d for d in sub_devs],
    )

    # Further partition each sub-device
    subsub_dev_eu_count = [
        [sd.max_compute_units for sd in d.create_sub_devices(partition=(1, 3))]
        for d in sub_devs
    ]
    print("Sub-sub-device #EU: ", subsub_dev_eu_count)


def subdivide_by_affinity(affinity="numa"):
    """
    Create sub-devices partitioning by affinity.
    """
    cpu_d = dpctl.SyclDevice("cpu")
    try:
        sub_devs = cpu_d.create_sub_devices(partition=affinity)
        print(
            "{0} sub-devices were created with respective "
            "#EUs being {1}".format(
                len(sub_devs), [d.max_compute_units for d in sub_devs]
            )
        )
    except Exception:
        print("Device partitioning by affinity was not successful.")


def create_subdevice_queue():
    """
    Partition a CPU sycl device into sub-devices.
    Create a multi-device sycl context.

    """
    cpu_d = dpctl.SyclDevice("cpu")
    cpu_count = cpu_d.max_compute_units
    sub_devs = cpu_d.create_sub_devices(partition=cpu_count // 2)
    multidevice_ctx = dpctl.SyclContext(sub_devs)
    # create a SyclQueue for each sub-device, using commont
    # multi-device context
    q0, q1 = [dpctl.SyclQueue(multidevice_ctx, d) for d in sub_devs]
    # for each sub-device allocate 26 bytes
    m0 = dpctl.memory.MemoryUSMDevice(26, queue=q0)
    m1 = dpctl.memory.MemoryUSMDevice(26, queue=q1)
    # populate m0 with host data of spaces
    hostmem = bytearray(b" " * 26)
    # copy spaces into m1
    m1.copy_from_host(hostmem)
    for i in range(26):
        hostmem[i] = ord("a") + i
    # copy character sequence into m0
    m0.copy_from_host(hostmem)
    # from from m0 to m1. Due to using multi-device context,
    # copying can be done directly
    m1.copy_from_device(m0)
    return bytes(m1.copy_to_host())


if __name__ == "__main__":
    import _runner as runner

    runner.run_examples(
        "Examples for working with subdevices in dpctl.", globals()
    )
