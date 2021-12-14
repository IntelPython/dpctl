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

"""Examples illustrating SYCL device selection using filter strings.
"""

import dpctl


def print_device(d):
    "Display information about given device argument."
    if type(d) is not dpctl.SyclDevice:
        raise ValueError
    print("Name: ", d.name)
    print("Vendor: ", d.vendor)
    print("Driver version: ", d.driver_version)
    print("Backend: ", d.backend)
    print("Max EU: ", d.max_compute_units)


def select_using_filter():
    """
    Demonstrate the usage of a filter string to create a SyclDevice.

    """
    try:
        d1 = dpctl.SyclDevice("cpu")
        print_device(d1)
    except ValueError:
        print("A CPU type device is not available on the system")

    try:
        d1 = dpctl.SyclDevice("opencl:cpu:0")
        print_device(d1)
    except ValueError:
        print("An OpenCL CPU driver needs to be installed on the system")

    d1 = dpctl.SyclDevice("0")
    print_device(d1)

    try:
        d1 = dpctl.SyclDevice("gpu")
        print_device(d1)
    except ValueError:
        print("A GPU type device is not available on the system")


if __name__ == "__main__":
    import _runner as runner

    runner.run_examples("Filter selection examples for dpctl.", globals())
