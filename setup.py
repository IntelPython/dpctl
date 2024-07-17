#               Data Parallel Control Library (dpctl)
#
# Copyright 2020-2024 Intel Corporation
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

import skbuild
import skbuild.setuptools_wrap
import skbuild.utils
import versioneer

skbuild.setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url="https://github.com/IntelPython/dpctl",
    packages=[
        "dpctl",
        "dpctl.memory",
        "dpctl.tensor",
        "dpctl.program",
        "dpctl.utils",
    ],
    package_data={
        "dpctl": [
            "tests/*.*",
            "tests/helper/*.py",
            "tests/elementwise/*.py",
            "tests/*.pyx",
            "tests/input_files/*",
            "resources/cmake/*.cmake",
            "include/*.h*",
            "include/syclinterface/*.h*",
            "include/syclinterface/Config/*.h",
            "include/syclinterface/Support/*.h",
            "tensor/libtensor/include/kernels/*.h*",
            "tensor/libtensor/include/kernels/sorting/*.h*",
            "tensor/libtensor/include/kernels/elementwise_functions/*.h*",
            "tensor/libtensor/include/kernels/linalg/*.h*",
            "tensor/libtensor/include/utils/*.h*",
            "tensor/include/dlpack/*.*",
            "include/dpctl/_sycl*.h",
            "include/dpctl/memory/_memory*.h",
            "include/dpctl/program/_program*.h",
            "include/dpctl/tensor/_usmarray*.h",
            "*.pxd",
            "memory/*.pxd",
            "tensor/*.pxd",
            "program/*.pxd",
        ]
    },
    include_package_data=False,
)
