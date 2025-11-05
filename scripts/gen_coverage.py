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
    capture_cmd_output,
    clean_build_dir,
    err,
    install_editable,
    log_cmake_args,
    make_cmake_args,
    resolve_compilers,
    run,
)


def find_bin_llvm(compiler):
    if os.path.isabs(compiler):
        bin_dir = os.path.dirname(compiler)
    else:
        compiler_path = capture_cmd_output(["which", compiler])
        if not compiler_path:
            raise RuntimeError(f"Compiler {compiler} not found in PATH")
        bin_dir = os.path.dirname(compiler_path)
    compiler_dir = os.path.join(bin_dir, "compiler")
    if os.path.exists(compiler_dir):
        bin_llvm = compiler_dir
    else:
        bin_dir = os.path.dirname(bin_dir)
        bin_llvm = os.path.join(bin_dir, "bin-llvm")
    if not os.path.exists(bin_llvm):
        raise RuntimeError(f"--bin-llvm value {bin_llvm} not found")
    return bin_llvm


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
        type=str,
        help="Path to GTestConfig.cmake file for a custom GTest installation",
    )
    p.add_argument(
        "--bin-llvm",
        type=str,
        help="Path to folder where llvm-cov/llvm-profdata can be found",
    )
    p.add_argument(
        "--skip-pytest",
        dest="run_pytest",
        action="store_false",
        help="Skip running pytest and coverage generation",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove build dir before rebuild (default: False)",
    )

    return p.parse_args()


def main():
    is_linux = "linux" in sys.platform
    if not is_linux:
        err(f"{sys.platform} not supported", "gen_coverage")
    args = parse_args()
    setup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    c_compiler, cxx_compiler = resolve_compilers(
        args.oneapi,
        args.c_compiler,
        args.cxx_compiler,
        args.compiler_root,
    )
    bin_llvm = find_bin_llvm(c_compiler)

    if args.clean:
        clean_build_dir(setup_dir)

    # Level Zero state (on unless explicitly disabled)
    level_zero_enabled = False if args.no_level_zero else True

    cmake_args = make_cmake_args(
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
        level_zero=level_zero_enabled,
        verbose=args.verbose,
    )

    cmake_args += ["-DDPCTL_GENERATE_COVERAGE=ON"]
    cmake_args += ["-DDPCTL_BUILD_CAPI_TESTS=ON"]
    cmake_args += [f"-DDPCTL_COVERAGE_REPORT_OUTPUT={setup_dir}"]

    if args.gtest_config:
        cmake_args += [f"-DCMAKE_PREFIX_PATH={args.gtest_config}"]

    env = os.environ.copy()

    if bin_llvm:
        env["PATH"] = ":".join((env.get("PATH", ""), bin_llvm))
        env["LLVM_TOOLS_HOME"] = bin_llvm

    log_cmake_args(cmake_args, "gen_coverage")

    build_extension(
        setup_dir,
        env,
        cmake_args,
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

    cmake_build_dir = capture_cmd_output(
        ["find", "_skbuild", "-name", "cmake-build"],
        cwd=setup_dir,
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

        def find_objects():
            objects = []
            sfx_regexp = sysconfig.get_config_var("EXT_SUFFIX").replace(
                ".", r"\."
            )
            regexp1 = re.compile(r"^_tensor_.*impl" + sfx_regexp)
            regexp2 = re.compile(r"^^_device_queries" + sfx_regexp)

            def is_py_ext(fn):
                return re.match(regexp1, fn) or re.match(regexp2, fn)

            for root, _, files in os.walk("dpctl"):
                for file in files:
                    if not file.endswith(".so"):
                        continue
                    if is_py_ext(file) or "DPCTLSyclInterface" in file:
                        objects.extend(["-object", os.path.join(root, file)])
            print("Using objects: ", objects)
            return objects

        objects = find_objects()
        instr_profile_fn = "dpctl_pytest.profdata"

        run(
            [
                os.path.join(bin_llvm, "llvm-profdata"),
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
                    os.path.join(bin_llvm, "llvm-cov"),
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
