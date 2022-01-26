import os
import subprocess
import sys


def run(
    use_oneapi=True,
    c_compiler=None,
    cxx_compiler=None,
    level_zero=True,
    compiler_root=None,
    cmake_executable=None,
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
    setup_dir = os.path.dirname(os.path.dirname(__file__))
    cmake_args = [
        sys.executable,
        "setup.py",
        "develop",
    ]
    if cmake_executable:
        cmake_args += [
            "--cmake-executable=" + cmake_executable,
        ]
    cmake_args += [
        "--",
        "-G",
        "Unix Makefiles",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DCMAKE_C_COMPILER:PATH=" + c_compiler,
        "-DCMAKE_CXX_COMPILER:PATH=" + cxx_compiler,
        "-DDPCTL_ENABLE_LO_PROGRAM_CREATION=" + ("ON" if level_zero else "OFF"),
        "-DDPCTL_DPCPP_FROM_ONEAPI:BOOL=" + ("ON" if use_oneapi else "OFF"),
    ]
    if compiler_root:
        cmake_args += [
            "-DDPCTL_DPCPP_HOME_DIR:PATH=" + compiler_root,
        ]
    subprocess.check_call(
        cmake_args, shell=False, cwd=setup_dir, env=os.environ
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Driver to build dpctl for in-place installation"
    )
    driver = parser.add_argument_group(title="Coverage driver arguments")
    driver.add_argument("--c-compiler", help="Name of C compiler", default=None)
    driver.add_argument(
        "--cxx-compiler", help="Name of C++ compiler", default=None
    )
    driver.add_argument(
        "--oneapi",
        help="Is one-API installation",
        dest="oneapi",
        action="store_true",
    )
    driver.add_argument(
        "--compiler-root", type=str, help="Path to compiler home directory"
    )
    driver.add_argument(
        "--cmake-executable", type=str, help="Path to cmake executable"
    )
    driver.add_argument(
        "--no-level-zero",
        help="Enable Level Zero support",
        dest="level_zero",
        action="store_false",
    )
    args = parser.parse_args()

    if args.oneapi:
        args.c_compiler = "icx"
        args.cxx_compiler = "icpx"
        args.compiler_root = None
    else:
        args_to_validate = [
            "c_compiler",
            "cxx_compiler",
            "compiler_root",
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
        cmake_executable=args.cmake_executable,
    )
