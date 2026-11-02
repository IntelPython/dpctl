# AGENTS.md

dpctl is a Python/Cython wrapper over a subset of SYCL, built on a C API
library (`libsyclinterface`). It is compiled with Intel oneAPI DPC++ (`icx`/`icpx`).

## Environment

- Activate the compiler before building or testing: `source /opt/intel/oneapi/setvars.sh`
  (or use a conda env with `dpcpp_linux-64`). The environment does not persist
  between shell invocations; source it in the same command:
  `source /opt/intel/oneapi/setvars.sh && python -m pytest dpctl/tests/...`.
- Check visible devices: `sycl-ls`. Tests skip on missing devices/aspects; a CPU
  OpenCL device is enough for most of the suite.
- Python build deps: `pip install numpy cython scikit-build cmake ninja setuptools pybind11 pytest "versioneer[toml]==0.29"`.

## Commands

- Build in place + editable install: `python scripts/build_locally.py --verbose`
  - defaults to `icx`/`icpx` on `PATH`; for other compilers pass absolute
    `--c-compiler`/`--cxx-compiler` paths, or names plus `--compiler-root=<bin dir>`
  - `--debug`, `--clean`, `--target-cuda[=sm_XX]`, `--target-hip=<arch>`,
    `--cmake-opts="-D..."`
  - the first configure clones the Level Zero headers and downloads pybind11;
    offline, pass `--no-level-zero` (or
    `--cmake-opts="-DDPCTL_LEVEL_ZERO_INCLUDE_DIR=<dir>"`) and install `pybind11`
- Python tests: `python -m pytest dpctl/tests`
  - one file: `python -m pytest dpctl/tests/test_sycl_queue.py`
  - one test: `python -m pytest dpctl/tests/test_sycl_queue_memset.py::test_memset_partial_count`
  - without a C compiler/Cython: add `-k "not test_cython_api and not test_c_headers"`
- C API tests (Linux; GoogleTest must be findable by CMake, e.g. via `GTEST_ROOT`):
  `(cd libsyclinterface && bash dbg_build.sh)`
- Coverage (Linux; C + Python, rebuilds in place with `-DDPCTL_BUILD_CAPI_TESTS=ON`,
  then runs the full pytest suite unless `--skip-pytest`): `python scripts/gen_coverage.py`,
  then `python scripts/build_locally.py --clean` for a normal build
- Docs (Linux only; needs Doxygen and the `docs` extras from `pyproject.toml`):
  `python scripts/gen_docs.py`, plus `--doxyrest-root=<path>` for the C API reference
- Lint/format one file (no build needed): `pre-commit run --files <path>`
- Lint/format everything: `pre-commit run --all-files`
  (black, isort, flake8, cython-lint, clang-format, codespell, bandit, shellcheck,
  gitleaks, actionlint). Config: `pyproject.toml`, `.flake8`, `.clang-format`.

## Structure

- `libsyclinterface/` — C API over SYCL (C++ implementation)
  - `include/syclinterface/*.h` — public C headers, `DPCTL<Type>_<Op>` functions
  - `source/*.cpp` — implementations
  - `helper/` — error handlers and enum/string conversion utils
  - `tests/*.cpp` — GoogleTest suite; `*.spv` are prebuilt kernel fixtures
- `dpctl/` — Python package
  - `_backend.pxd` — Cython `extern` declarations of the C API
  - `_sycl_*.pyx/.pxd` — `SyclDevice`, `SyclQueue`, `SyclContext`, `SyclEvent`, ...
  - `memory/` — USM memory objects (`MemoryUSMDevice`, ...), IPC memory
  - `compiler/` — `SyclKernelBundle` creation (SPIR-V, SYCL source)
  - `program/` — deprecated alias of `dpctl.compiler`
  - `utils/` — order manager, device queries, Intel device info
  - `apis/include/` — Python C-API headers for extensions (`dpctl_capi.h`, `dpctl4pybind11.hpp`)
  - `tests/` — pytest suite; `tests/helper/` has `get_queue_or_skip` and friends
- `examples/` — python, cython, pybind11, c extension examples (each with README)
- `docs/doc_sources/` — Sphinx sources
- `conda-recipe/`, `conda-recipe-cf/` — conda packaging used by CI
- `scripts/` — build, coverage, docs drivers
- `benchmarks/` — asv benchmarks
- `cmake/` — CMake config for downstream projects (`dpctl-config.cmake`)

## Adding a SYCL feature (typical PR shape)

1. Declare in `libsyclinterface/include/syclinterface/dpctl_sycl_<x>_interface.h`
   with a Doxygen comment and ownership attributes. For an existing SYCL type,
   extend its files in steps 1-4 rather than creating new ones.
2. Implement in `libsyclinterface/source/dpctl_sycl_<x>_interface.cpp`.
3. Add GoogleTest cases in `libsyclinterface/tests/test_sycl_<x>_interface.cpp`.
4. Declare in `dpctl/_backend.pxd`; wrap in `dpctl/_sycl_<x>.pyx` (and `.pxd` if
   the method is `cdef` or `cpdef`).
5. Add pytest coverage in `dpctl/tests/`.
6. Export new public names (e.g. enums) in `dpctl/__init__.py` and list them in
   `docs/doc_sources/api_reference/dpctl/*.rst`.
7. For user-visible changes, add a `CHANGELOG.md` entry (see Git).

Reference PRs to copy from:
- New queue operation: gh-2361 (`DPCTLQueue_MemsetWithEvents` →
  `SyclQueue.memset`/`memset_async`, `dpctl/tests/test_sycl_queue_memset.py`).
- New device info query: gh-2324 (`dpctl_sycl_device_interface.h` →
  `dpctl/_sycl_device.pyx`; add a `check_<name>(device)` to `list_of_checks` in
  `dpctl/tests/_device_attributes_checks.py`). New enums go through
  `dpctl_sycl_enum_types.h`, `libsyclinterface/helper/`, and `dpctl/enum_types.py`.

## Conventions

### C API
- Opaque refs (`DPCTLSyclQueueRef`, ...) via `wrap<T>`/`unwrap<T>`
  (`dpctl_sycl_type_casters.hpp`).
- Annotate ownership: `__dpctl_give` (caller must delete), `__dpctl_take`
  (callee consumes), `__dpctl_keep` (borrowed). See `Support/MemOwnershipAttrs.h`.
- Never let a C++ exception cross the C boundary: catch, call
  `error_handler(e, __FILE__, __func__, __LINE__)`, return `nullptr`/error value.
- Null-check every incoming ref and report via `error_handler`.
- Errors are silent unless `DPCTL_VERBOSITY` is set; use `warning` when debugging
  a `nullptr` return (`error` hides the null-ref messages).

### Cython / Python
- Release the GIL (`with nogil:`) around calls that may block on the device.
- Python-visible errors: raise specific exceptions (`TypeError`, `ValueError`,
  `SyclQueueCreationError`, ...) when the C API returns `NULL`.
- Free-threaded CPython (3.14t) is supported; do not rely on the GIL for
  thread safety of shared state. New `.pyx` files need
  `# cython: freethreading_compatible = True`, like the existing ones.
- Line length 80 (black, isort, cython-lint, clang-format).

### Tests
- Create queues with `get_queue_or_skip()` (`dpctl/tests/helper`) or
  `try: dpctl.SyclQueue()` / `except dpctl.SyclQueueCreationError: pytest.skip(...)`;
  skip on missing aspects, never assume a GPU or fp64 support.
- Test both sync and `*_async` variants when adding queue operations.

### Files
- New files need the Apache-2.0 license header from `CONTRIBUTING.md` with the
  current year; all C/C++ files (not only headers) also need the `\file` Doxygen block.
- Comments and docstrings: short, describe what the code does.

## Git

- Branch off `master`; pre-commit blocks commits directly to `master` and
  `maintenance/*`.
- One concern per PR; fill in `.github/pull_request_template.md`.
- Update `CHANGELOG.md` under `## [dev]` (Keep a Changelog sections: Added, Changed,
  Deprecated, Removed, Fixed, Maintenance) for user-visible changes, e.g.
  `* Added ... [gh-NNNN](https://github.com/IntelPython/dpctl/pull/NNNN)`; the
  number is only known once the PR is opened.

## Boundaries

- Generated, do not edit: `dpctl/*.cxx`, `dpctl/**/*.cxx`, `dpctl/include/`
  (headers copied/generated at build time), `_skbuild/`, `build/`.
- `dpctl/_version.py` is a committed, vendored versioneer file; do not edit it by hand.
- Edit C headers in `libsyclinterface/include/` and Python C-API headers in
  `dpctl/apis/include/`, not their copies under `dpctl/include/`.
  `Config/dpctl_config.h` is generated from `dpctl_config.h.in`; edit the `.in`.
- In-place `.so` files survive branch switches; rebuild (`--clean`) before
  trusting test results after checking out another branch.
- Changing an existing C API signature or enum value is a breaking change:
  call it out in the PR and in a `**NOTE**: This release contains breaking changes:`
  list at the top of the `[dev]` section (as under 0.23.0).
- `.github/workflows/` actions are pinned to commit SHAs (checked by zizmor);
  keep that when adding steps.
- Don't change `conda-recipe*/`, release versions, or `docs/doc_versions.txt`
  unless asked; call such changes out in the PR.
