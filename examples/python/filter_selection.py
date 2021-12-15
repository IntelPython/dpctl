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


def select_using_filter():
    """
    Demonstrate the usage of a filter string to create a SyclDevice.

    """
    try:
        d1 = dpctl.SyclDevice("cpu")
        d1.print_device_info()
    except ValueError:
        print("A CPU type device is not available on the system")

    try:
        d1 = dpctl.SyclDevice("opencl:cpu:0")
        d1.print_device_info()
    except ValueError:
        print("An OpenCL CPU driver needs to be installed on the system")

    d1 = dpctl.SyclDevice("0")
    d1.print_device_info()

    try:
        d1 = dpctl.SyclDevice("gpu")
        d1.print_device_info()
    except ValueError:
        print("A GPU type device is not available on the system")


if __name__ == "__main__":
    import _runner as runner

    runner.run_examples("Filter selection examples for dpctl.", globals())
