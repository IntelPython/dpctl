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

import argparse
import os
import re
import subprocess
import sys
import sysconfig

# add scripts dir to Python path so we can import _build_helper
sys.path.insert(0, os.path.abspath("scripts"))

from _build_helper import (  # noqa: E402
    build_extension,
    clean_build_dir,
    err,
    install_editable,
    make_cmake_args,
    resolve_compilers,
    run,
    warn,
)


def parse_args():
    p = argparse.ArgumentParser(description="Build dpctl and generate coverage")

    p.add_argument(
        "--c-compiler", default=None, help="Path or name of C compiler"
    )
    p.add_argument(
        "--cxx-compiler", default=None, help="Path or name of C++ compiler"
    )
    p.add_argument(
        "--compiler-root",
        type=str,
        default=None,
        help="Path to compiler installation root",
    )
    p.add_argument(
        "--oneapi",
        dest="oneapi",
        action="store_true",
        help="Use default oneAPI compiler layout",
    )

    p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose makefile output",
    )

    p.add_argument(
        "--no-level-zero",
        dest="no_level_zero",
        action="store_true",
        default=False,
        help="Disable Level Zero backend (deprecated: use --target-level-zero "
        "OFF)",
    )
    p.add_argument(
        "--target-level-zero",
        action="store_true",
        help="Enable Level Zero backend explicitly",
    )

    p.add_argument(
        "--generator", type=str, default="Ninja", help="CMake generator"
    )
    p.add_argument(
        "--cmake-executable",
        type=str,
        default=None,
        help="Path to CMake executable used by build",
    )

    p.add_argument(
        "--cmake-opts",
        type=str,
        default="",
        help="Additional options to pass directly to CMake",
    )

    p.add_argument(
        "--gtest-config",
        help="Path to GTestConfig.cmake file for a custom GTest installation",
    )
    p.add_argument(
        "--bin-llvm",
        help="Path to folder where llvm-cov/llvm-profdata can be found",
    )
    p.add_argument("--skip-pytest", dest="run_pytest", action="store_false")
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove build dir before rebuild (default: False)",
    )

    return p.parse_args()


def find_objects(setup_dir):
    objects = []
    sfx_regexp = sysconfig.get_config_var("EXT_SUFFIX").replace(".", r"\.")
    regexp1 = re.compile(r"^_tensor_.*impl" + sfx_regexp)
    regexp2 = re.compile(r"^^_device_queries" + sfx_regexp)

    def is_py_ext(fn):
        return re.match(regexp1, fn) or re.match(regexp2, fn)

    for root, _, files in os.walk(os.path.join(setup_dir, "dpctl")):
        for file in files:
            if not file.endswith(".so"):
                continue
            if is_py_ext(file) or "DPCTLSyclInterface" in file:
                objects.extend(["-object", os.path.join(root, file)])
            print("[gen_coverage] Using objects:", objects)
            return objects


def main():
    is_linux = "linux" in sys.platform
    if not is_linux:
        err(f"{sys.platform} not supported")
    args = parse_args()
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    c_compiler, cxx_compiler = resolve_compilers(
        args.oneapi, args.c_compiler, args.cxx_compiler, args.compiler_root
    )

    if args.clean:
        clean_build_dir(setup_dir)

    if args.no_level_zero and args.target_level_zero:
        err("Cannot combine --no-level-zero and --target-level-zero")

    # Level Zero state (on unless explicitly disabled)
    if args.no_level_zero:
        level_zero_enabled = False
    elif args.target_level_zero:
        level_zero_enabled = True
    else:
        level_zero_enabled = True

    cmake_args = make_cmake_args(
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        level_zero=level_zero_enabled,
        verbose=args.verbose,
    )

    cmake_args += " -DDPCTL_GENERATE_COVERAGE=ON"
    cmake_args += " -DDPCTL_BUILD_CAPI_TESTS=ON"
    cmake_args += f" -DDPCTL_COVERAGE_REPORT_OUTPUT={setup_dir}"

    if args.gtest_config:
        cmake_args += " -DCMAKE_PREFIX_PATH={args.gtest_config}"

    env = os.environ.copy()

    if "CMAKE_ARGS" in env and env["CMAKE_ARGS"].strip():
        warn("Ignoring pre-existing CMAKE_ARGS in environment")
        del env["CMAKE_ARGS"]

    if args.bin_llvm:
        env["PATH"] = ":".join((env.get("PATH", ""), args.bin_llvm))
        env["LLVM_TOOLS_HOME"] = args.bin_llvm
        llvm_profdata = os.path.join(args.bin_llvm, "llvm-profdata")
        llvm_cov = os.path.join(args.bin_llvm, "llvm-cov")
        cmake_args += f" -DLLVM_TOOLS_HOME={args.bin_llvm}"
        cmake_args += f" -DLLVM_PROFDATA={llvm_profdata}"
        cmake_args += f" -DLLVM_COV={llvm_cov}"
        # Add LLVMCov_EXE for CMake find_package(LLVMCov)
        cmake_args += f" -DLLVMCov_EXE={llvm_cov}"

    print(f"[gen_coverage] Using CMake args:\n {env['CMAKE_ARGS']}")

    env["CMAKE_ARGS"] = cmake_args

    build_extension(
        setup_dir,
        env,
        cmake_executable=args.cmake_executable,
        generator=args.generator,
        build_type="Coverage",
    )
    install_editable(setup_dir, env)

    cmake_build_dir = (
        subprocess.check_output(
            ["find", "_skbuild", "-name", "cmake-build"], cwd=setup_dir
        )
        .decode("utf-8")
        .strip("\n")
    )
    print(f"[gen_coverage] Found CMake build dir: {cmake_build_dir}")

    run(
        ["cmake", "--build", ".", "--target", "llvm-cov-report"],
        cwd=cmake_build_dir,
    )

    if args.run_pytest:
        env["LLVM_PROFILE_FILE"] = "dpctl_pytest.profraw"
        pytest_cmd = [
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
            "--no-sycl-interface-test",
        ]
        run(pytest_cmd, env=env, cwd=setup_dir)

        objects = find_objects(setup_dir)
        instr_profile_fn = "dpctl_pytest.profdata"

        run(
            [
                os.path.join(args.bin_llvm or "", "llvm-profdata"),
                "merge",
                "-sparse",
                env["LLVM_PROFILE_FILE"],
                "-o",
                instr_profile_fn,
            ]
        )

        with open("dpctl_pytest.lcov", "w") as fh:
            subprocess.check_call(
                [
                    os.path.join(args.bin_llvm or "", "llvm-cov"),
                    "export",
                    "-format=lcov",
                    "-ignore-filename-regex=/tmp/icpx*",
                    f"-instr-profile={instr_profile_fn}",
                ]
                + objects,
                cwd=setup_dir,
                env=env,
                stdout=fh,
            )
        print("[gen_coverage] Coverage export complete: dpctl_pytest.lcov")
    else:
        print(
            "[gen_coverage] Skipping pytest and coverage collection "
            "(--skip-pytest)"
        )

    print("[gen_coverage] Done")


if __name__ == "__main__":
    main()
