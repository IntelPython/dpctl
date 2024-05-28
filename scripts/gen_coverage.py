#                      Data Parallel Control (dpctl)
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

import os
import re
import subprocess
import sys
import sysconfig


def run(
    use_oneapi=True,
    c_compiler=None,
    cxx_compiler=None,
    level_zero=True,
    compiler_root=None,
    run_pytest=False,
    bin_llvm=None,
    gtest_config=None,
    verbose=False,
):
    IS_LIN = False

    if "linux" in sys.platform:
        IS_LIN = True
    elif sys.platform in ["win32", "cygwin"]:
        pass
    else:
        assert False, sys.platform + " not supported"

    if not IS_LIN:
        raise RuntimeError(
            "This scripts only supports coverage collection on Linux"
        )
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmake_args = [
        sys.executable,
        "setup.py",
        "develop",
        "--build-type=Coverage",
        "--generator=Ninja",
        "--",
        "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
        "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
        "-DDPCTL_ENABLE_L0_PROGRAM_CREATION=" + ("ON" if level_zero else "OFF"),
        "-DDPCTL_GENERATE_COVERAGE=ON",
        "-DDPCTL_BUILD_CAPI_TESTS=ON",
        "-DDPCTL_COVERAGE_REPORT_OUTPUT_DIR=" + setup_dir,
    ]
    env = dict()
    if bin_llvm:
        env = {
            "PATH": ":".join((os.environ.get("PATH", ""), bin_llvm)),
            "LLVM_TOOLS_HOME": bin_llvm,
        }
        env.update({k: v for k, v in os.environ.items() if k != "PATH"})
    if gtest_config:
        cmake_args += ["-DCMAKE_PREFIX_PATH=" + gtest_config]
    if verbose:
        cmake_args += [
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
        ]
    subprocess.check_call(cmake_args, shell=False, cwd=setup_dir, env=env)
    cmake_build_dir = (
        subprocess.check_output(
            ["find", "_skbuild", "-name", "cmake-build"], cwd=setup_dir
        )
        .decode("utf-8")
        .strip("\n")
    )
    subprocess.check_call(
        ["cmake", "--build", ".", "--target", "llvm-cov-report"],
        cwd=cmake_build_dir,
    )
    env["LLVM_PROFILE_FILE"] = "dpctl_pytest.profraw"
    subprocess.check_call(
        [
            "pytest",
            "-q",
            "-ra",
            "--disable-warnings",
            "--cov-config",
            "pyproject.toml",
            "--cov",
            "dpctl",
            "--cov-report",
            "term-missing",
            "--pyargs",
            "dpctl",
            "-vv",
            "--ignore=dpctl/tensor/libtensor/tests",
        ],
        cwd=setup_dir,
        shell=False,
        env=env,
    )

    def find_objects():
        import os

        objects = []
        sfx_regexp = sysconfig.get_config_var("EXT_SUFFIX").replace(".", r"\.")
        regexp1 = re.compile(r"^_tensor_.*impl" + sfx_regexp)
        regexp2 = re.compile(r"^^_device_queries" + sfx_regexp)

        def is_py_ext(fn):
            return re.match(regexp1, fn) or re.match(regexp2, fn)

        for root, _, files in os.walk("dpctl"):
            for file in files:
                if not file.endswith(".so"):
                    continue
                if is_py_ext(file) or file.find("DPCTLSyclInterface") != -1:
                    objects.extend(["-object", os.path.join(root, file)])
        print("Using objects: ", objects)
        return objects

    objects = find_objects()
    instr_profile_fn = "dpctl_pytest.profdata"
    # generate instrumentation profile data
    subprocess.check_call(
        [
            os.path.join(bin_llvm, "llvm-profdata"),
            "merge",
            "-sparse",
            env["LLVM_PROFILE_FILE"],
            "-o",
            instr_profile_fn,
        ]
    )
    # export lcov
    with open("dpctl_pytest.lcov", "w") as fh:
        subprocess.check_call(
            [
                os.path.join(bin_llvm, "llvm-cov"),
                "export",
                "-format=lcov",
                "-ignore-filename-regex=/tmp/icpx*",
                "-instr-profile=" + instr_profile_fn,
            ]
            + objects,
            stdout=fh,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Driver to build dpctl and generate coverage"
    )
    driver = parser.add_argument_group(title="Coverage driver arguments")
    driver.add_argument("--c-compiler", help="Name of C compiler", default=None)
    driver.add_argument(
        "--cxx-compiler", help="Name of C++ compiler", default=None
    )
    driver.add_argument(
        "--not-oneapi",
        help="Is one-API installation",
        dest="oneapi",
        action="store_false",
    )
    driver.add_argument(
        "--compiler-root", type=str, help="Path to compiler home directory"
    )
    driver.add_argument(
        "--no-level-zero",
        help="Enable Level Zero support",
        dest="level_zero",
        action="store_false",
    )
    driver.add_argument(
        "--skip-pytest",
        help="Run pytest and collect coverage",
        dest="run_pytest",
        action="store_false",
    )
    driver.add_argument(
        "--bin-llvm", help="Path to folder where llvm-cov can be found"
    )
    driver.add_argument(
        "--verbose",
        help="Build using vebose makefile mode",
        dest="verbose",
        action="store_true",
    )
    driver.add_argument(
        "--gtest-config",
        help="Path to the GTestConfig.cmake file to locate a "
        + "custom GTest installation.",
    )
    args = parser.parse_args()

    if args.oneapi:
        args.c_compiler = "icx"
        args.cxx_compiler = "icpx"
        args.compiler_root = None
        icx_path = subprocess.check_output(["which", "icx"])
        bin_dir = os.path.dirname(icx_path)
        compiler_dir = os.path.join(bin_dir.decode("utf-8"), "compiler")
        if os.path.exists(compiler_dir):
            args.bin_llvm = os.path.join(bin_dir.decode("utf-8"), "compiler")
        else:
            bin_dir = os.path.dirname(bin_dir)
            args.bin_llvm = os.path.join(bin_dir.decode("utf-8"), "bin-llvm")
        assert os.path.exists(args.bin_llvm)
    else:
        args_to_validate = [
            "c_compiler",
            "cxx_compiler",
            "compiler_root",
            "bin_llvm",
        ]
        for p in args_to_validate:
            arg = getattr(args, p, None)
            if not isinstance(arg, str):
                opt_name = p.replace("_", "-")
                raise RuntimeError(
                    f"Option {opt_name} must be provided is "
                    "using non-default DPC++ layout"
                )
            if not os.path.exists(arg):
                raise RuntimeError(f"Path {arg} must exist")

    run(
        use_oneapi=args.oneapi,
        c_compiler=args.c_compiler,
        cxx_compiler=args.cxx_compiler,
        level_zero=args.level_zero,
        compiler_root=args.compiler_root,
        run_pytest=args.run_pytest,
        bin_llvm=args.bin_llvm,
        gtest_config=args.gtest_config,
        verbose=args.verbose,
    )
