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

"""Demonstrates SYCL queue selection operations provided by dpctl.
"""

import dpctl


def print_available_platforms():
    """
    Print information about SYCL platforms visible to runtime.

    Environment variable `SYCL_DEVICE_FILTER` affects this list.
    """
    dpctl.lsplatform()


def list_available_platforms():
    """
    Get a list of SyclPlatform instances corresponding to platforms
    visible to SYCL runtime.

    Environment variable `SYCL_DEVICE_FILTER` affects this list.
    """
    for p in dpctl.get_platforms():
        print(p)


if __name__ == "__main__":
    import _runner as runner

    runner.run_examples("", globals())
