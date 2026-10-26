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
- Python build deps: `pip install numpy cython scikit-build cmake ninja pytest "versioneer[toml]==0.29"`.

## Commands

- Build in place + editable install: `python scripts/build_locally.py --verbose`
  - defaults to `icx`/`icpx` on `PATH`; other compilers need `--compiler-root`
  - `--debug`, `--clean`, `--target-cuda[=sm_XX]`, `--target-hip=<arch>`,
    `--cmake-opts="-D..."`
- Python tests: `python -m pytest dpctl/tests`
  - one file: `python -m pytest dpctl/tests/test_sycl_queue.py`
  - one test: `python -m pytest dpctl/tests/test_sycl_queue.py -k memset`
  - without a C compiler/Cython: add `-k "not test_cython_api and not test_c_headers"`
- C API tests (GoogleTest, needs `GTEST_ROOT`): `bash libsyclinterface/dbg_build.sh`
  (runs `make check`)
- Coverage (C + Python, rebuilds with `-DDPCTL_BUILD_CAPI_TESTS=ON`):
  `python scripts/gen_coverage.py --oneapi`
- Docs (needs doxyrest): `python scripts/gen_docs.py --doxyrest-root=<path>`
- Lint/format one file (no build needed): `pre-commit run --files <path>`
- Lint/format everything: `pre-commit run --all-files`
  (black, isort, flake8, cython-lint, clang-format, codespell, bandit, shellcheck,
  gitleaks, actionlint). Config: `pyproject.toml`, `.flake8`, `.clang-format`.

## Structure

- `libsyclinterface/` — C API over SYCL (C++ implementation)
  - `include/syclinterface/*.h` — public C headers, `DPCTL<Type>_<Op>` functions
  - `source/*.cpp` — implementations
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

## Adding a SYCL feature (typical PR shape)

1. Declare in `libsyclinterface/include/syclinterface/dpctl_sycl_<x>_interface.h`
   with a Doxygen comment and ownership attributes.
2. Implement in `libsyclinterface/source/dpctl_sycl_<x>_interface.cpp`.
3. Add GoogleTest cases in `libsyclinterface/tests/test_sycl_<x>_interface.cpp`.
4. Declare in `dpctl/_backend.pxd`; wrap in `dpctl/_sycl_<x>.pyx` (and `.pxd` if
   the method is `cdef`).
5. Add pytest coverage in `dpctl/tests/`.
6. Add a `CHANGELOG.md` entry with the PR link.

Reference PRs to copy from:
- New queue operation: gh-2361 (`DPCTLQueue_Memset` → `SyclQueue.memset`,
  `dpctl/tests/test_sycl_queue_memset.py`).
- New device info query: gh-2324 (`dpctl_sycl_device_interface.h` →
  `dpctl/_sycl_device.pyx`; register the check in
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
- Errors are silent unless `DPCTL_VERBOSITY=warning` (or `error`) is set; set it
  when debugging a `nullptr` return.

### Cython / Python
- Release the GIL (`with nogil:`) around calls that may block on the device.
- Python-visible errors: raise specific exceptions (`TypeError`, `ValueError`,
  `SyclQueueCreationError`, ...) when the C API returns `NULL`.
- Free-threaded CPython (3.14t) is supported; do not rely on the GIL for
  thread safety of shared state.
- Line length 80 (black, isort, cython-lint, clang-format).

### Tests
- Get devices with `get_queue_or_skip()` / skip on missing aspects; never assume
  a GPU or fp64 support.
- Test both sync and `*_async` variants when adding queue operations.

### Files
- New files need the Apache-2.0 license header from `CONTRIBUTING.md`
  (C/C++ headers also need the `\file` Doxygen block).
- Comments and docstrings: short, describe what the code does.

## Git

- Branch off `master`; pre-commit blocks commits directly to `master` and
  `maintenance/*`.
- One concern per PR; fill in `.github/pull_request_template.md`.
- Update `CHANGELOG.md` (Keep a Changelog sections: Added, Changed, Deprecated,
  Removed, Fixed, Maintenance) for user-visible changes, citing `gh-NNNN`.

## Boundaries

- Generated, do not edit: `dpctl/*.cxx`, `dpctl/**/*.cxx`, `dpctl/include/`
  (headers copied/generated at build time), `dpctl/_version.py` (versioneer),
  `_skbuild/`, `build/`.
- Edit C headers in `libsyclinterface/include/`, not their copies under `dpctl/include/`.
- In-place `.so` files survive branch switches; rebuild (`--clean`) before
  trusting test results after checking out another branch.
- Changing an existing C API signature or enum value is a breaking change:
  call it out in the PR and the CHANGELOG `NOTE` list.
- `.github/workflows/` actions are pinned to commit SHAs (checked by zizmor);
  keep that when adding steps.
- `conda-recipe*/`, release versions, and doc version lists need maintainer review.
